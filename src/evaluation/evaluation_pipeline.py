import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from src.evaluation import get_gemini_model, get_metrics


def build_test_cases(rag, df):
    """
    Build a list of test cases for evaluating a Retrieval-Augmented Generation (RAG) model.

    Args:
        rag: An instance of the RAG model, which provides a `query` method to generate answers
             and a `vector_store.retriever().get_relevant_documents` method to retrieve relevant contexts.
        df (pd.DataFrame): A DataFrame containing the test data. Each row should include:
            - "input": The input query for the model.
            - "expected_output" (optional): The expected output for the query.
            - "contexts" (optional): Predefined contexts to be used for the test case.

    Returns:
        list: A list of `LLMTestCase` objects, each representing a test case with
          the following attributes:
            - `input`: The input query.
            - `actual_output`: The model's generated answer.
            - `expected_output`: The expected output for the query.
            - `contexts` (if supported): The predefined contexts for the test case.
            - `retrieval_contexts` (if supported): The contexts retrieved by the model.
            - `additional_metadata` (if `contexts` and `retrieval_contexts`
              are not directly supported):
              A dictionary containing the predefined and retrieved contexts.
    """

    test_cases = []

    for _, row in df.iterrows():
        q = row["input"]
        expected = row.get("expected_output", "")
        ctx_text = row.get("contexts", "")
        contexts = [ctx_text] if isinstance(ctx_text, str) and ctx_text else []

        # Get model answer + actual retrieved contexts
        answer = rag.query(q)
        retrieved_docs = rag.vector_store.retriever().get_relevant_documents(q)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # Build test case
        case = LLMTestCase(
            input=q,
            actual_output=answer,
            expected_output=expected,
        )

        # Attach contexts for newer DeepEval versions
        if hasattr(case, "contexts"):
            case.contexts = contexts
            case.retrieval_contexts = retrieved_contexts
        else:
            case.additional_metadata = {
                "contexts": contexts,
                "retrieval_contexts": retrieved_contexts,
            }

        test_cases.append(case)
    return test_cases


def run_evaluation(rag, eval_data_path):
    """
    Executes the evaluation process for the given RAG (Retrieval-Augmented Generation)
      model.

    Args:
        rag: The RAG model instance to be evaluated.
        eval_data_path (str): The file path to the evaluation data in CSV format.

    Returns:
        dict: A dictionary containing the evaluation results, including detailed metrics.

    This function performs the following steps:
    1. Loads the evaluation data from the specified CSV file.
    2. Builds test cases using the provided RAG model and the loaded data.
    3. Retrieves the Gemini model and associated evaluation metrics.
    4. Runs the evaluation process using the test cases and metrics.
    5. Returns the evaluation results, including detailed information.
    """
    # Load evaluation data
    df = pd.read_csv(eval_data_path)

    # Build test cases
    test_cases = build_test_cases(rag, df)

    # Get Gemini model and metrics
    gemini_model = get_gemini_model()
    metrics = get_metrics(model=gemini_model)

    # Run evaluation
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        return_details=True,
    )

    return results
