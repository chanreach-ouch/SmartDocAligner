import pytest
import os
import sys
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_unwarping.paddle_uvdoc import unwarp_with_paddle
from document_unwarping.onnx_uvdoc import unwarp_with_onnx

@pytest.fixture
def mock_image(tmp_path):
    # Create a dummy image
    img_path = tmp_path / "test_img.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

@pytest.fixture
def mock_gray_image(tmp_path):
    img_path = tmp_path / "test_gray.jpg"
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

@pytest.fixture
def mock_onnx_model_path(tmp_path):
    model_path = tmp_path / "dummy.onnx"
    model_path.write_text("dummy")
    return str(model_path)

# Mocks for Paddle
@patch('document_unwarping.paddle_uvdoc.get_paddle_model')
def test_paddle_invalid_path(mock_get_model):
    res = unwarp_with_paddle("invalid.jpg", "out.jpg")
    assert res["error"] is not None
    assert "Input file not found" in res["error"]

@patch('document_unwarping.paddle_uvdoc.get_paddle_model')
def test_paddle_success(mock_get_model, mock_image, tmp_path):
    mock_model = MagicMock()
    mock_res = MagicMock()
    mock_res.image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_model.predict.return_value = [mock_res]
    mock_get_model.return_value = mock_model
    
    out_path = str(tmp_path / "out.jpg")
    res = unwarp_with_paddle(mock_image, out_path)
    
    assert res["error"] is None
    assert res["width"] == 50
    assert res["height"] == 50
    assert os.path.exists(out_path)

# Mocks for ONNX
@patch('document_unwarping.onnx_uvdoc.get_onnx_session')
def test_onnx_invalid_path(mock_get_session):
    res = unwarp_with_onnx("invalid.jpg", "out.jpg", "invalid.onnx")
    assert res["error"] is not None
    assert "Input file not found" in res["error"]

def test_onnx_missing_model(mock_image):
    res = unwarp_with_onnx(mock_image, "out.jpg", "missing.onnx")
    assert res["error"] is not None
    assert "Model not found" in res["error"]

@patch('document_unwarping.onnx_uvdoc.get_onnx_session')
def test_onnx_success_grid(mock_get_session, mock_image, mock_onnx_model_path, tmp_path):
    mock_session = MagicMock()
    
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 512, 512]
    
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.shape = [1, 2, 512, 512]
    
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    
    # Mock return grid
    grid = np.zeros((1, 2, 512, 512), dtype=np.float32)
    mock_session.run.return_value = [grid]
    
    mock_get_session.return_value = mock_session
    
    out_path = str(tmp_path / "out_onnx.jpg")
    res = unwarp_with_onnx(mock_image, out_path, mock_onnx_model_path)
    
    assert res["error"] is None
    assert res["width"] == 100 # Remap keeps original size
    assert res["height"] == 100
    assert os.path.exists(out_path)

@patch('document_unwarping.onnx_uvdoc.get_onnx_session')
def test_onnx_success_image(mock_get_session, mock_image, mock_onnx_model_path, tmp_path):
    mock_session = MagicMock()
    
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 512, 512]
    
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.shape = [1, 3, 512, 512]
    
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    
    # Mock return image
    out_img = np.zeros((1, 3, 512, 512), dtype=np.float32)
    mock_session.run.return_value = [out_img]
    
    mock_get_session.return_value = mock_session
    
    out_path = str(tmp_path / "out_onnx2.jpg")
    res = unwarp_with_onnx(mock_image, out_path, mock_onnx_model_path)
    
    assert res["error"] is None
    assert res["width"] == 512
    assert res["height"] == 512
    assert os.path.exists(out_path)
