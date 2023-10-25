import abc


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False, echo=False):
        self.multiline = multiline
        self.echo = echo

    def prompt_for_input(self, role) -> str:
        if not self.multiline:
            inputs = input(f"{role}: ")
        else:
            prompt_data = []
            line = input(f"{role} [ctrl-d/z on empty line to end]: ")
            while True:
                prompt_data.append(line.strip())
                try:
                    line = input()
                except EOFError as e:
                    break
            inputs = "\n".join(prompt_data)
        if self.echo:
            print(f'{role}: {inputs}')
        return inputs

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def output(self, message):
        output = message['text']
        print(output, flush=True)
        return output


class DistChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self.multiline = multiline
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def _prompt_for_input(self, role) -> str:
        if not self.multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_input(self, role):
        if self.rank == 0:
            inputs = self._prompt_for_input(role)
        else:
            inputs = ""
        inputs = self.comm.bcast(inputs, root=0)
        return inputs

    def prompt_for_output(self, role: str):
        self._print(f"{role}: ", end="", flush=True)

    def _print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                self._print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        self._print(" ".join(output_text[pre:]), flush=True)
        outputs = " ".join(output_text)
        return outputs

    def output(self, message):
        output = message['text']
        self._print(output, flush=True)
        return output


