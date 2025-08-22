from common_utils.generative_ai.prompt.vlm.base import VideoSummaryChatPrompt
from common_utils.generative_ai.prompt.base import BaseChatPrompt
class DeliverySummaryChatPrompt(VideoSummaryChatPrompt):
    system_template_str = (
        VideoSummaryChatPrompt.system_template_str + 
        "You are observing a delivery for a client at a specific location,"
        "tracking relevant actions and events."
    )
    user_template_str = (
        "For our client {tenant} at this location {location} you're going to observe a delivery and summarize the delivery concisely"
    )

class DeliverySummaryChatPrompt(VideoSummaryChatPrompt):
    user_template_str = (
        "If available, incorporate the previous analysis:\n{input}\n\n"
        "Now analyze the new incoming frames from the delivery video.\n"
        "1. Describe the sequence of events in clear chronological order."
        "The frames contain in the bottom left corner a timestamp use that.\n"
        "2. Identify all objects being unloaded or dumped.\n"
        "3. Highlight anything unusual, unexpected, or noteworthy.\n"
        "Focus on factual, observable details and avoid speculation."
    )

    input_variables = ["input"]



class DeliveryFrameWithJSONChatPrompt(VideoSummaryChatPrompt):
    system_template_str = (
        VideoSummaryChatPrompt.system_template_str
        + "\nYou will provide the information in a structured JSON format."
    )
    user_template_str = (
        "Observe the delivery in these video frames and describe the sequence of events. "
        "Identify the objects being unloaded or dumped, and note any unusual or unexpected activity. "
        "Return your output as a structured JSON following this template:\n{json_template}"
    )
    input_variables = ["json_template"]



    