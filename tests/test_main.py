import unittest
from unittest.mock import patch, MagicMock, call
from fastapi.testclient import TestClient

import numpy as np
import os
import cv2
import sys
import json
import io

import app.predict
import app.preproc_inference
import project_name.data.preprocessing
import app_file


class TestLoadModel(unittest.TestCase):

    def setUp(self):
        app.predict._model = None

    @patch("app.predict.tf.keras.models.load_model")
    def test_load_model_success(self, mock_load_model: unittest.mock):
        # testcase for the function executing correctly

        # create a fake Keras model and let the mock method return it
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model


        # pretend to call the actual function, which should return a mock_model
        model = app.predict.load_model()

        # assert that the tested method is a mock_model and that it is not trainable
        self.assertIs(model, mock_model)
        self.assertFalse(model.trainable)

        # assert that the function was called (once) with the right arguments
        mock_load_model.assert_called_once_with(app.predict.MODEL_PATH)

    # make the mock method return an exception and prevent the process from exiting because of it
    @patch("app.predict.tf.keras.models.load_model", side_effect=Exception("mocked error"))
    @patch("app.predict.sys.exit")
    def test_load_model_failure(self, mock_exit, mock_load_model):
        # testcase for the function returning an exception

        # fake print method to check that printing happens without actually printing anything
        with patch("builtins.print") as mock_print:
            # call the mocked function and assert it was called correctly
            app.predict.load_model()
            mock_load_model.assert_called_once_with(app.predict.MODEL_PATH)

            # assert that the function tried to print something
            mock_print.assert_called()

            # check that the correct text ends up in the exception
            args, _ = mock_print.call_args
            self.assertIn("Error loading model", args[0])

            # assert that sys.exit was called
            mock_exit.assert_called_once_with(1)


class TestLoadAndNormalize(unittest.TestCase):

    @patch("app.predict.kimage.img_to_array")
    @patch("app.predict.kimage.load_img")
    def test_load_and_normalize(self, mock_load_img, mock_img_to_array):

        # create a fake nonsense image
        fake_img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        mock_img_to_array.return_value = fake_img_array

        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        result = app.predict.load_and_normalize("fake/path/to/image.jpg")

        # check whether the mocked functions are called correctly
        mock_load_img.assert_called_once_with("fake/path/to/image.jpg", target_size=(128, 128))
        mock_img_to_array.assert_called_once_with(mock_img)

        # assert that the shape of the fake image is right
        self.assertEqual(result.shape, (1, 128, 128, 3))

        # assert that the fake image was normalized properly
        self.assertTrue(np.all(result >= 0.0) and np.all(result <= 1.0))
        self.assertAlmostEqual(result.max(), fake_img_array.max() / 255.0, places=5)


class TestPredictOnImage(unittest.TestCase):

    @patch("app.predict.load_and_normalize")
    def test_predict_on_image_real(self, mock_load_and_normalize):
        # Create fake input image (batch of one)
        mock_image_array = np.random.rand(1, 128, 128, 3)
        mock_load_and_normalize.return_value = mock_image_array

        # Mock model and prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.9]])  # class 1 = "real"

        result = app.predict.predict_on_image(mock_model, "fake/path.jpg")

        mock_load_and_normalize.assert_called_once_with("fake/path.jpg")
        mock_model.predict.assert_called_once_with(mock_image_array, verbose=0)

        self.assertEqual(result["label"], "real")
        self.assertAlmostEqual(result["confidence"], 0.9)

    @patch("app.predict.load_and_normalize")
    def test_predict_on_image_fake(self, mock_load_and_normalize):
        mock_image_array = np.random.rand(1, 128, 128, 3)
        mock_load_and_normalize.return_value = mock_image_array

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.8, 0.2]])  # class 0 = "fake"

        result = app.predict.predict_on_image(mock_model, "fake/path.jpg")

        self.assertEqual(result["label"], "fake")
        self.assertAlmostEqual(result["confidence"], 0.8)


class TestNormalizeMissingDir(unittest.TestCase):

    @patch("app.predict.os.path.exists", return_value=True)
    def test_path_exists(self, mock_exists):
        path = "fake/path/image_preprocessed"
        result = app.predict.normalize_missing_dir(path)
        self.assertEqual(result, path)
        mock_exists.assert_called_once_with(path)

    @patch("app.predict.os.path.exists", return_value=False)
    @patch("app.predict.os.path.isdir", return_value=True)
    def test_jpg_preprocessed_alt_exists(self, mock_isdir, mock_exists):
        path = "fake/image.jpg_preprocessed"
        expected_alt = "fake/image_preprocessed"
        result = app.predict.normalize_missing_dir(path)
        self.assertEqual(result, expected_alt)
        mock_isdir.assert_called_once_with(expected_alt)

    @patch("app.predict.os.path.exists", return_value=False)
    @patch("app.predict.os.path.isdir", return_value=True)
    def test_jpeg_preprocessed_alt_exists(self, mock_isdir, mock_exists):
        path = "fake/image.jpeg_preprocessed"
        expected_alt = "fake/image_preprocessed"
        result = app.predict.normalize_missing_dir(path)
        self.assertEqual(result, expected_alt)
        mock_isdir.assert_called_once_with(expected_alt)

    @patch("app.predict.os.path.exists", return_value=False)
    @patch("app.predict.os.path.isdir", return_value=False)
    def test_alt_does_not_exist(self, mock_isdir, mock_exists):
        path = "fake/image.jpg_preprocessed"
        result = app.predict.normalize_missing_dir(path)
        self.assertEqual(result, path)
        mock_isdir.assert_called_once_with("fake/image_preprocessed")

    @patch("app.predict.os.path.exists", return_value=False)
    def test_path_with_no_suffix(self, mock_exists):
        path = "fake/image.png"
        result = app.predict.normalize_missing_dir(path)
        self.assertEqual(result, path)
        mock_exists.assert_called_once_with(path)


class TestFaceDetection(unittest.TestCase):

    @patch("app.preproc_inference.detect_face")
    @patch("app.preproc_inference.cv2.cvtColor")
    @patch("app.preproc_inference.cv2.imwrite")
    @patch("app.preproc_inference.os.makedirs")
    @patch("builtins.print")
    def test_face_detected(
        self, mock_print, mock_makedirs, mock_imwrite, mock_cvtcolor, mock_detect_face
    ):

        fake_face = np.ones((128, 128, 3), dtype=np.uint8)
        mock_detect_face.return_value = fake_face
        mock_cvtcolor.return_value = fake_face

        image_path = "/path/to/fake/image.jpg"
        base = "/path/to/fake/image"
        out_dir = f"{base}_preprocessed"
        save_path = os.path.join(out_dir, "image_preprocessed.jpg")

        app.preproc_inference.process_single(image_path)

        mock_detect_face.assert_called_once_with(image_path, 10, (128, 128))
        mock_makedirs.assert_called_once_with(out_dir, exist_ok=True)
        mock_cvtcolor.assert_called_once_with(fake_face, cv2.COLOR_RGB2BGR)
        mock_imwrite.assert_called_once_with(save_path, fake_face)
        mock_print.assert_called_with(f"Saved {save_path}")

    @patch("app.preproc_inference.sys.exit", side_effect=SystemExit)
    @patch("builtins.print")
    @patch("app.preproc_inference.detect_face", return_value=None)
    @patch("app.preproc_inference.os.makedirs")
    def test_no_face_detected(self, mock_makedirs, mock_detect_face, mock_print, mock_exit):
        image_path = "fake/image.jpg"

        with self.assertRaises(SystemExit):
            app.preproc_inference.process_single(image_path)

        mock_print.assert_called_with(f"Failed to preprocess {image_path}", file=sys.stderr)
        mock_exit.assert_called_once_with(2)
        mock_makedirs.assert_not_called()


class TestPreprocessing(unittest.TestCase):

    @patch("app.preproc_inference.sys.exit", side_effect=SystemExit)
    @patch("app.preproc_inference.cv2.cvtColor")
    @patch("app.preproc_inference.cv2.imwrite")
    @patch("app.preproc_inference.detect_face", return_value=None)
    @patch("app.preproc_inference.os.makedirs")
    @patch("app.preproc_inference.os.listdir")
    @patch("builtins.print")
    def test_process_images_without_faces(
            self, mock_print, mock_listdir, mock_makedirs,
            mock_detect_face, mock_imwrite, mock_cvtcolor,
            mock_sys_exit
    ):
        folder = "fake/no/faces"
        image_files = ["x.jpg", "y.jpeg"]
        mock_listdir.return_value = image_files

        with self.assertRaises(SystemExit):
            app.preproc_inference.process_folder(folder)

        mock_imwrite.assert_not_called()
        self.assertEqual(mock_print.call_count, 3)
        mock_print.assert_has_calls([
            call(f"Skipping {f}: no face", file=sys.stderr) for f in image_files
        ], any_order=True)

    @patch("app.preproc_inference.detector")
    def test_align_face_no_faces(self, mock_detector):
        dummy_img = np.random.randint(0, 255, (128,128,3), dtype=np.uint8)

        mock_detector.return_value = []

        aligned_img = project_name.data.preprocessing.align_face(dummy_img)

        np.testing.assert_array_equal(aligned_img, dummy_img)

client = TestClient(app_file.app)

class TestUploadImageAPI(unittest.TestCase):

    @patch("app_file.subprocess.run")
    def test_upload_successful(self, mock_run):

        preproc_result = MagicMock(returncode=0, stdout='', stderr='')
        prediction_output = json.dumps({"label": "real", "confidence": 0.87})
        predict_result = MagicMock(returncode=0, stdout=prediction_output, stderr='')

        mock_run.side_effect = [preproc_result, predict_result]

        file_data = io.BytesIO(b"fake jpeg data")
        response = client.post(
                               "/upload-image/",
                               files={"file": ("test.jpg", file_data, "fake_image/jpeg")}
                              )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["image_is_real"])
        self.assertAlmostEqual(data["confidence"], 0.87)

    @patch("app_file.subprocess.run")
    def test_upload_no_face_detected(self, mock_run):
         preproc_result = MagicMock(returncode=2, stderr="No face found")
         mock_run.return_value = preproc_result

         file_data = io.BytesIO(b"fake jpeg data")
         response = client.post(
                               "/upload-image/",
                               files={"file": ("noface.jpg", file_data, "fake_image/jpeg")}
                              )

         self.assertEqual(response.status_code, 400)
         self.assertIn("No face detected", response.text)

    @patch("app_file.subprocess.run")
    def test_upload_preprocessing_failure(self, mock_run):
        preproc_result = MagicMock(returncode=1, stderr="preprocessing failed")
        mock_run.return_value = preproc_result

        file_data = io.BytesIO(b"fake jpeg data")
        response = client.post(
                               "/upload-image/",
                               files={"file": ("fail.jpg", file_data, "fake_image/jpeg")}
                              )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Pre-processing error", response.text)



if __name__ == '__main__':
    unittest.main(verbosity=2)