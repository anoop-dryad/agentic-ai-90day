from google.genai import types


def broken_tool():
    raise ValueError("simulated crash")


broken_tool_declaration = types.FunctionDeclaration(
    name="broken_tool",
    description="A tool that always fails. Use this when the user says 'test failure'.",
    parameters={
        "type": "object",
        "properties": {},
    },
)
