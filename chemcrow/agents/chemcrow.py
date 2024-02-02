import langchain
from pydantic import ValidationError, BaseModel, Field
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from dotenv import load_dotenv
from typing import Optional
import openai

from .prompts import *
from .tools import make_tools, make_safety_tools


def _make_llm(model, temp, api_key, streaming: bool = False):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key = api_key
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key = api_key
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm

class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4",
        tools_model="gpt-4",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {}
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming)
        except ValidationError:
            raise ValueError('Invalid OpenAI API key')

        if tools is None:
            api_keys['OPENAI_API_KEY'] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(
                tools_llm,
                api_keys = api_keys,
                verbose=verbose
            )

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs['output']


class ReduceTask(BaseModel):
    task: str = Field(description = "Describe what task is the user specifically asking for. What is the main thing user is asking our system to execute.")
    compound: Optional[str] = Field(description = 'If the user specifies any particular substance, which one is it. Substances may be specified with common names, iupac notation, smiles, or cas numbers.')

    @classmethod
    def from_userquery(cls, query):
        task = client.chat.completions.create(
            response_model=cls,
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a safety guard. Your task is to check what the intentions of users are.",
                },
                {
                    "role": "user",
                    "content": f"Here is the query from the user: {query}",
                },
            ],
            max_retries=3,
        )
        return task

    def to_query(self):
        return f"{self.task} {self.compound}"


class SafetyCrow:
        def __init__(
                    self,
                    tools=None,
                    model="gpt-4",
                    tools_model="gpt-4",
                    temp=0.1,
                    max_iterations=40,
                    verbose=True,
                    streaming: bool = True,
                    openai_api_key: Optional[str] = None,
                    api_keys: dict = {}
                    ):
            

            load_dotenv()
            try:
                self.llm = _make_llm(model, temp, openai_api_key, streaming)
            except ValidationError:
                raise ValueError('Invalid OpenAI API key')

            if tools is None:
                api_keys['OPENAI_API_KEY'] = openai_api_key
                tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
                tools = make_safety_tools(
                    tools_llm,
                    api_keys = api_keys,
                    verbose=verbose
                )

            # Initialize agent
            self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
                tools=tools,
                agent=ChatZeroShotAgent.from_llm_and_tools(
                    self.llm,
                    tools,
                    suffix=SUFFIX,
                    format_instructions=SAFETY_FORMAT_INSTRUCTIONS,
                    question_prompt=SAFETY_QUESTIONS,
                ),
                verbose=True,
                max_iterations=max_iterations,
            )

            rephrase = PromptTemplate(
                input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
            )

            self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

        def run(self, query):
            red_task = ReduceTask.from_userquery(query)
            new_q = red_task.to_query()

            outputs = self.agent_executor({"input": new_q})
            return outputs['output']
