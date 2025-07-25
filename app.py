from model import AzureOpenAIClient
import json
import requests
import os
from pypdf import PdfReader
import gradio as gr

def notify(text):
    requests.post(
        "https://ntfy.sh/agentone",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    notify(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}


def record_improper_question(question):
    notify(f"Recording {question}")
    return {"recorded": "ok"}



record_improper_question_json = {
    "name": "record_improper_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_improper_question_json},
]


class Company:
    def __init__(self):
        self.client = AzureOpenAIClient()
        self.name = "AgentOne"
        reader = PdfReader("docs/profile.pdf")
        self.profile = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.profile += text
        with open("docs/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s founder, company, and products. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name} founder's background and profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, or find violence/vulgar in question use your record_improper_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Profile:\n{self.profile}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def sanitize_message(self, msg):
        return {"role": msg["role"], "content": msg["content"]}


    def chat(self, message, history):
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + [self.sanitize_message(m) for m in history]
            + [{"role": "user", "content": message}]
        )
        done = False
        while not done:
            response = self.client.generate_response(
                messages, tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    company = Company()
    gr.ChatInterface(company.chat, type="messages").launch()