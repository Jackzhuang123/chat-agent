core/__init__.py:15:def register_read_only_tool(tool_name: str) -> None:
core/__init__.py:22:class ParallelConfig:
core/__init__.py:29:class EnhancedParallelExecutor:
core/agent_middlewares.py:112:class SkillsContextMiddleware(AgentMiddleware):
core/agent_middlewares.py:12:class AgentMiddleware(ABC):
core/agent_middlewares.py:140:class UploadedFilesMiddleware(AgentMiddleware):
core/agent_middlewares.py:172:class ToolResultGuardMiddleware(AgentMiddleware):
core/agent_middlewares.py:245:class ConversationSummaryMiddleware(AgentMiddleware):
core/agent_middlewares.py:30:def _inject_context_before_last_user(messages: List[Dict[str, str]], context_message: Dict[str, str]) -> List[Dict[str, str]]:
core/agent_middlewares.py:354:class ContextWindowMiddleware(AgentMiddleware):
core/agent_middlewares.py:40:class _OnceInjectMiddleware(AgentMiddleware):
core/agent_middlewares.py:439:class CompletenessMiddleware(_OnceInjectMiddleware):
core/agent_middlewares.py:468:class AskUserQuestionMiddleware(AgentMiddleware):
core/agent_middlewares.py:509:class CompletionStatusMiddleware(AgentMiddleware):
core/agent_middlewares.py:550:class SearchBeforeBuildingMiddleware(_OnceInjectMiddleware):
core/agent_middlewares.py:584:class RepoOwnershipMiddleware(_OnceInjectMiddleware):
core/agent_middlewares.py:61:class RuntimeModeMiddleware(AgentMiddleware):
core/agent_middlewares.py:81:class PlanModeMiddleware(AgentMiddleware):
core/agent_skills.py:26:class SkillManager:
core/agent_skills.py:301:class SkillInjector:
core/agent_skills.py:397:def create_example_skills(skills_dir: str = None):
core/agent_tools.py:1095:class ToolRegistry:
core/agent_tools.py:1129:def create_web_search_tool_placeholder() -> Dict[str, Any]:
core/agent_tools.py:15:class ToolExecutor:
core/agent_tools.py:668:class ToolParser:
core/components/completion_guard.py:10:def looks_finished(response: str, session: "SessionContext", runtime_context: Dict = None) -> bool:
core/components/format_corrector.py:22:def inject_format_correction(messages: List[Dict], response: str, work_dir: str = "", custom_msg: str = None) -> List[Dict]:
core/components/format_corrector.py:9:def should_retry_tool_format(response: str, has_successful_tool_result: bool = False) -> bool:
core/components/loop_detector.py:13:def detect_loop(session: "SessionContext", max_same: int = 3) -> bool:
core/components/output_cleaner.py:19:def clean_react_tags(text: str) -> str:
core/components/output_cleaner.py:6:def strip_trailing_tool_call(text: str) -> str:
core/components/task_injector.py:12:def inject_task_context(
core/context_retriever.py:19:class ContextRetriever:
core/langgraph_agent.py:100:def _make_json_serializable(obj: Any) -> Any:
core/langgraph_agent.py:125:def _sanitize_state_update(update: Dict[str, Any]) -> Dict[str, Any]:
core/langgraph_agent.py:130:def _should_skip_rag_for_file_request(task: str) -> bool:
core/langgraph_agent.py:142:def _normalize_run_mode(run_mode: str, plan_mode: bool = False) -> str:
core/langgraph_agent.py:153:def _append_unique_fact(facts: List[Dict[str, Any]], fact: Dict[str, Any], max_items: int = 12) -> None:
core/langgraph_agent.py:162:def _update_facts_ledger(session: SessionContext, tool_name: str, tool_args: Dict[str, Any], result_obj: Dict[str, Any], success: bool) -> None:
core/langgraph_agent.py:207:def _summarize_facts_ledger(session: SessionContext) -> Dict[str, int]:
core/langgraph_agent.py:20:def _is_alive_patch(self) -> bool:
core/langgraph_agent.py:217:def session_to_state(session: SessionContext) -> Dict[str, Any]:
core/langgraph_agent.py:228:def state_to_session(state_data: Dict[str, Any], session: SessionContext) -> None:
core/langgraph_agent.py:250:class AgentNode:
core/langgraph_agent.py:446:class ToolNode:
core/langgraph_agent.py:57:class AgentState(TypedDict, total=False):
core/langgraph_agent.py:602:class FinalizeNode:
core/langgraph_agent.py:643:class LoopDetectedNode:
core/langgraph_agent.py:659:class LangGraphAgent:
core/langgraph_agent.py:77:def _safe_model_call(model_forward_fn: Callable, messages: list,
core/model_forward.py:30:def create_qwen_model_forward(qwen_agent, system_prompt_base: str = ""):
core/model_forward.py:7:def _merge_system_messages(messages: List[Dict[str, str]], combined_system_prompt: str = "") -> List[Dict[str, str]]:
core/monitor_logger.py:100:def get_monitor_logger() -> logging.Logger:
core/monitor_logger.py:108:def make_trace_id(prefix: str = "req") -> str:
core/monitor_logger.py:113:def set_log_level(level: str):
core/monitor_logger.py:127:def log_execution_time(func: Optional[Callable] = None, *, level: int = logging.INFO):
core/monitor_logger.py:158:def log_async_execution_time(func: Optional[Callable] = None, *, level: int = logging.INFO):
core/monitor_logger.py:214:def info(msg: str, *args, **kwargs):
core/monitor_logger.py:218:def warning(msg: str, *args, **kwargs):
core/monitor_logger.py:222:def error(msg: str, *args, **kwargs):
core/monitor_logger.py:226:def debug(msg: str, *args, **kwargs):
core/monitor_logger.py:230:def exception(msg: str, *args, **kwargs):
core/monitor_logger.py:235:def log_event(event_type: str, message: str, level: int = logging.INFO, **extra_fields):
core/monitor_logger.py:249:def log_startup(app_name: str = "QwenAgent", port: Optional[int] = None):
core/monitor_logger.py:257:def log_shutdown(app_name: str = "QwenAgent"):
core/monitor_logger.py:262:def log_http_request(method: str, path: str, status: int, duration_ms: float):
core/monitor_logger.py:266:def log_function_call(level: int = logging.DEBUG):
core/monitor_logger.py:292:def log_async_function_call(level: int = logging.DEBUG):
core/monitor_logger.py:36:def _setup_logger() -> logging.Logger:
core/monitor_logger.py:81:class _ColoredFormatter(logging.Formatter):
core/multi_agent.py:123:class ExecutorAgent:
core/multi_agent.py:187:class ReviewerAgent:
core/multi_agent.py:228:class MultiAgentOrchestrator:
core/multi_agent.py:23:def _safe_model_call(model_forward_fn: Callable, messages: list, system_prompt: str = "", **kwargs) -> str:
core/multi_agent.py:342:class ReActMultiAgentOrchestrator:
core/multi_agent.py:46:class PlannerAgent:
core/prompts.py:266:def get_system_prompt(mode: str, **kwargs) -> str:
core/prompts.py:295:def inject_few_shot_examples(messages: list, tool_name: str = None, max_examples: int = 2) -> list:
core/rag_intent_router.py:21:class IntentType(Enum):
core/rag_intent_router.py:32:class IntentResult:
core/rag_intent_router.py:40:class RAGIntentRouter:
core/reflection.py:11:class EnhancedReflectionEngine:
core/state_manager.py:10:class SessionContext:
core/state_manager.py:31:class WorkflowStateManager:
core/streaming_framework.py:145:def create_streaming_wrapper(framework) -> StreamingFramework:
core/streaming_framework.py:16:class StreamEvent:
core/streaming_framework.py:30:class StreamingFramework:
core/tool_enforcement_middleware.py:189:class DirectCommandDetector:
core/tool_enforcement_middleware.py:25:class ToolEnforcementMiddleware(AgentMiddleware):
core/tool_learner.py:12:class ToolUsagePattern:
core/tool_learner.py:28:class AdaptiveToolLearner:
core/tool_learner.py:369:class ContextFeatureExtractor:
core/vector_memory.py:143:class VectorMemory:
core/vector_memory.py:30:class MemoryEntry:
core/vector_memory.py:52:class EmbeddingProvider:
core/vector_memory.py:59:class LocalEmbeddingProvider(EmbeddingProvider):
