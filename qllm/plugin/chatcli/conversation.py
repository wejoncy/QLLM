import dataclasses
from typing import List

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if message:
          	  if i == 0:
          		  ret += self.system + message
          	  else:
          		  ret += role + " " + message + seps[i % 2]
            else:
          	  ret += role
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

CONV_MAP = {
    'llama2': Conversation(
        name="llama-2",
        system="",
        roles=("[INST]", "[/INST]"),
        messages=(),
        offset=0,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2], 
    )
}

def get_conv(name: str):
	assert name in CONV_MAP, f'not find name in conversations.'
	return CONV_MAP[name].copy()
