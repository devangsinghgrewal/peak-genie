import json
import logging
from typing import Literal

from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.prompts import PromptTemplate

from info_gpt import constants
from info_gpt.agent import Agent
from info_gpt.db import get_db_with_embedding


def load_model(
    model_type: Literal["GPT4All", "OpenAI", "LlamaCpp"] = "OpenAI",
) -> BaseRetrievalQA:
    logging.info("Loading model...")
    vector_store = get_db_with_embedding()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    if model_type == "GPT4All":
        llm_model = GPT4All(
            model=constants.MODEL_PATH,
            n_ctx=constants.MODEL_N_CTX,
            backend="gptj",
            verbose=True,
            n_threads=8,
            f16_kv=False,
            temp=0.15,
            echo=True,
            use_mlock=True,
            n_batch=16,
            allow_download=True,
        )
    if model_type == "LlamaCpp":
        llm_model = LlamaCpp(
            model=constants.MODEL_PATH,
            n_ctx=constants.MODEL_N_CTX,
        )
    elif model_type == "OpenAI":
        llm_model = OpenAI(temperature=constants.TEMPERATURE)
    else:
        error = "Model type not supported."
        raise Exception(error)

    return RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )


def make_query(query):
    template = """
    Format the answer in the following way-
    FYI:- will have some Additional Info about the subject asked .Reply in a humorous way ,witty way.
    Following is the query:
    {query}
    """

    prompt = PromptTemplate(input_variables=["query"], template=template)

    return prompt.format(query=query)


def ask(query, retrieval_chain, *, show_on_webapp=False):
    logging.info(f"Getting answer for the query: {query}")
    if "workflow" in query.lower() and "create" in query.lower():
        logging.info("Using the workflow creationg agent.")
        response = Agent().workflow_agent(query)
        return f"Workflow create with id: {json.loads(response.content)['id']}"

    query = "Give the answer in most readable format \n" + query
    result = retrieval_chain({"query": query})

    return f"{result['result']}"
