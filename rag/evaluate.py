# define evaluator
import nest_asyncio
import pandas as pd
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llama_dataset import download_llama_dataset

from rag.index_manager import IndexManager

nest_asyncio.apply()

# Refer to https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/

pd.set_option("display.max_colwidth", 0)
DATA_DIR = "../data"
llm = Settings.llm


# download dataset
# evaluator_dataset, _ = download_llama_dataset(
#     "MiniMtBenchSingleGradingDataset", "./mini_mt_bench_data"
# )
async def evaluate():
    # Start evaluators
    faithfulness = FaithfulnessEvaluator(llm=llm)
    relevancy = RelevancyEvaluator(llm=llm)
    correctness = CorrectnessEvaluator(llm=llm)

    # Get vector index
    base_index = IndexManager().base_index

    # Generate questions
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    dataset_generator = DatasetGenerator.from_documents(documents, llm=llm)
    qas = dataset_generator.generate_dataset_from_nodes(num=3)

    # Create evaluation runner
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness, "relevancy": relevancy},
        workers=8,
    )
    eval_results = await runner.aevaluate_queries(
        base_index.as_query_engine(llm=llm), queries=qas.questions
    )

    return eval_results


def inspect_results(key, results) -> float:
    """
    Inspect the results of the evaluation.
    :param key: faithfulness, relevancy, correctness
    :param results: eval results
    :return: score
    """
    print(results[key][0].dict().keys())
    print(f"---> {key} Results (Passing): {results[key][0].passing}")
    print(f"---> {key} Results (Response):\n{results[key][0].response}")
    print(f"---> {key} Results (Contexts):\n{results[key][0].contexts}")

    correct = 0
    for result in results[key]:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"---> {key} Score: {score}")
    return score


if __name__ == "__main__":
    import asyncio

    results = asyncio.run(evaluate())

    print(f"Available evaluation metrics: {results.keys()}")
    inspect_results("faithfulness", results)
    inspect_results("relevancy", results)
