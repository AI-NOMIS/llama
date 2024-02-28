from llama import Llama


class InferlessPythonModel:
    def initialize(self):
        pass

    def infer(self, inputs):
        generator = Llama.build(
            ckpt_dir="/var/nfs-mount/weigts-volume/llama2/llama-2-7b-chat/",
            tokenizer_path="tokenizer.model",
            max_seq_len=512,
            max_batch_size=6,
        )
        dialogs = [
            {
                "role": "user",
                "content": inputs.get("content", "what is the recipe of mayonnaise?"),
            }
        ]
        results = generator.chat_completion(
            dialogs,
            max_gen_len=inputs.get("max_gen_len", None),
            temperature=inputs.get("temperature", 0.6),
            top_p=inputs.get("top_p", 0.9),
        )

        return {"output": results[0]["generation"]["content"]}

    def finalize(self):
        pass
