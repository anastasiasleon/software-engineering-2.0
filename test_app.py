import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch
from app import classify_image, load_model

# Мок для модели и экстрактора признаков
@pytest.fixture
def mock_model_components():
    mock_feature_extractor = MagicMock()
    mock_model = MagicMock()

    # Настройка мока для feature_extractor
    mock_feature_extractor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    # Настройка мока для model.config.id2label
    mock_model.config.id2label = {0: "cat", 1: "dog"}

    # Настройка мока для model.__call__ (или model.forward)
    mock_logits = torch.tensor([[0.1, 0.9]]) # Предполагаем, что класс 1 (dog) имеет более высокую вероятность
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs

    return mock_feature_extractor, mock_model

def test_classify_image_mocked(mock_model_components):
    mock_feature_extractor, mock_model = mock_model_components

    with patch('app.feature_extractor', new=mock_feature_extractor):
        with patch('app.model', new=mock_model):
            # Создаем фиктивное изображение
            dummy_image = Image.new('RGB', (224, 224), color = 'red')

            # Выполняем классификацию
            predicted_label = classify_image(dummy_image)

            # Проверяем, что функция вернула ожидаемый результат
            assert predicted_label == "dog"

            # Проверяем, что экстрактор признаков был вызван
            mock_feature_extractor.assert_called_once_with(images=dummy_image, return_tensors="pt")

            # Проверяем, что модель была вызвана
            mock_model.assert_called_once()
