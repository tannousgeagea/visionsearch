from common_utils.generative_ai.prompt.base import BaseChatPrompt

class ImageDescriptionChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that accurately describes images "
        "without speculating about unseen details."
    )
    user_template_str = "Provide a comprehensive description of this image.\n{input}"


class ImageObjectDetectionChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that identifies and lists all visible objects "
        "in an image accurately."
    )
    user_template_str = "Focus on identifying and listing all objects visible in this image.\n{input}"


class ImageTextExtractionChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that accurately extracts and transcribes "
        "all visible text from images."
    )
    user_template_str = "Extract and transcribe all text visible in this image.\n{input}"


class ImageVisualQAChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that answers questions using only information "
        "visible in the image, without assuming unseen details."
    )
    user_template_str = "Answer the question based on what you can observe in this image.\n{input}"


class ImageSceneUnderstandingChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that analyzes images and describes "
        "the overall scene, setting, and context accurately."
    )
    user_template_str = "Analyze and describe the overall scene, setting, and context.\n{input}"


class ComparativeAnalysisChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that analyzes images and compares them, "
        "noting similarities and differences accurately."
    )
    user_template_str = "Compare and analyze the provided images, noting similarities and differences.\n{input}"


class ImageStyleAnalysisChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that analyzes images, focusing on artistic style, "
        "composition, and visual elements."
    )
    user_template_str = "Analyze the artistic style, composition, and visual elements.\n{input}"


class ImageContentModerationChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that analyzes images for potentially inappropriate "
        "or harmful content, providing accurate and responsible assessments."
    )
    user_template_str = "Analyze this image for any potentially inappropriate or harmful content.\n{input}"


class ImageJSONFormatGenerationPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant specialized in generating natural-language captions for waste bunker images, " \
        "using structured JSON-like outputs that describe objects and their attributes."
    )
    user_template_str = ""


class ImageCaptionFromJSONChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that generates concise and informative captions "
        "focusing on the overall scene based on provided JSON data."
    )
    user_template_str = (
        "Return a concise, informative caption focusing on the overall scene using the JSON: {json}. "
        "The image captured at {tenant} located at {location} contains the information in the JSON. "
        "In your response, provide only a natural language caption."
    )
    input_variables = ["tenant", "location", "json"]

class VideoSummaryChatPrompt(BaseChatPrompt):
    system_template_str = (
        "You are an AI assistant that analyzes video frames located at either the waste bunker or at a gate with a dropchute. Each frame you analyze "
        "you're able to pick up temporal relationships "
        "and context across each frame."
    )
    user_template_str = (
        "Analyze these video frames and provide a concise summary of their content."
    )


