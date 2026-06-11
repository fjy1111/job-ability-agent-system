class LLMCallError(RuntimeError):
    """统一给前端展示的大模型调用失败错误。"""

    def __init__(self) -> None:
        super().__init__("调用LLM失败")
