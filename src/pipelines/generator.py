from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMEngine:
    """
    LLMEngine is a class that provides an interface for interacting with a
      large language model (LLM) to generate text based on a given prompt.
      It uses the Hugging Face Transformers library to load
      a pre-trained model and tokenizer, and sets up a text generation pipeline.

    Attributes:
        model_id (str): The identifier of the pre-trained model to use.
          Defaults to "tiiuae/falcon-7b-instruct".
        device (str): The device to run the model on.
          Defaults to "auto", which automatically selects the device.

    Methods:
        __init__(model_id="tiiuae/falcon-7b-instruct", device="auto")
        generate(prompt: str, max_new_tokens: int = 150, temperature: float = 0,
          do_sample: bool = False)
    """

    def __init__(self, model_id="tiiuae/falcon-7b-instruct", device="auto"):
        """
        Initializes the generator with a specified model and device configuration.

        Args:
            model_id (str): The identifier of the pre-trained model to use. 
                Defaults to "tiiuae/falcon-7b-instruct".
            device (str): The device to load the model on. Can be "auto", "cpu", 
                or a specific GPU identifier. Defaults to "auto".

        Attributes:
            model_id (str): The identifier of the pre-trained model.
            device (str): The device configuration for the model.
            tokenizer (AutoTokenizer): The tokenizer loaded from the pre-trained model.
            model (AutoModelForCausalLM): The causal language model loaded
              from the pre-trained model.
            llm_pipeline (Pipeline): The text generation pipeline initialized
              with the model and tokenizer.
        """
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        self.llm_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0,
        do_sample: bool = False,
    ):
        """
        Generates a response based on the given prompt using the language model pipeline.

        Args:
            prompt (str): The input text prompt to generate a response for.
            max_new_tokens (int, optional): The maximum number of new tokens
              to generate. Defaults to 150.
            temperature (float, optional): Sampling temperature for controlling
              randomness. A lower value makes the output more deterministic.
              Defaults to 0.
            do_sample (bool, optional): Whether to use sampling for generation. 
                If False, the model uses greedy decoding. Defaults to False.

        Returns:
            str: The generated response text, with the prompt removed
              and leading/trailing whitespace stripped.
        """
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )[0]["generated_text"]
        response = response.replace(prompt, "").strip()
        return response
