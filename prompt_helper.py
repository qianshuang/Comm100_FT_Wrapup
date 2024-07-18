# -*- coding: utf-8 -*


instruction_template = """# GOAL #
Carefully analyze the CHAT HISTORY below between customer and customer service, summarize and sort out the values of the 2 fields of Category and Case_Status.
Let's work this out in a step by step way to be sure we have the right answer.

# CHAT HISTORY #
{}

# RETURN AS A JSON #
{{"Category": "The correct Category to which the case in the CHAT HISTORY should be assigned. If it cannot be categorized, leave it blank.", "Case_Status": "The final Case Status in the CHAT HISTORY. If it cannot be determined, leave it blank."}}"""

sys_message = "You are an trustworthy AI assistant that is very good at summarizing and organizing CHAT HISTORY between customer and customer service."
