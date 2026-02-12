from pydantic import BaseModel, Field


class ControlParam(BaseModel):
    """A parameter for a PL/SQL function."""
    name: str = Field(description="Parameter name, e.g. V_FBROK")
    direction: str = Field(default="IN", description="Parameter direction: IN, OUT, IN OUT")
    type: str = Field(default="CHAR", description="PL/SQL type: CHAR, VARCHAR2, NUMBER, etc.")
    description: str = Field(default="", description="Description of the parameter")


class Variable(BaseModel):
    """A local variable declaration in the PL/SQL function."""
    declaration: str = Field(description="Full variable declaration, e.g. V_COUNT NUMBER := 0;")


class Control(BaseModel):
    """A validation control/check within the function."""
    code: str = Field(description="Control code, e.g. NCD01249")
    description: str = Field(description="Natural-language description of the control")
    short_desc: str = Field(default="", description="Short description for NTC_DM_CTRL INSERT")
    error_code: str = Field(default="", description="Error code, e.g. NED01249")
    long_desc: str = Field(default="", description="Long error description for NTC_DM_CTRL INSERT")
    logic: str = Field(
        default="-- TODO: IMPLEMENT VALIDATION LOGIC\n    NULL;",
        description="PL/SQL logic block for this control"
    )


class FunctionSpec(BaseModel):
    """Complete specification for generating one PL/SQL function."""
    function_name: str = Field(description="Function name, e.g. NFD_CTRL_FBROK")
    function_description: str = Field(default="", description="Description of what the function does")
    parameters: list[ControlParam] = Field(default_factory=list)
    controls: list[Control] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Result of reviewing a generated SQL file."""
    function_name: str
    status: str = Field(description="PASS or FAIL")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
