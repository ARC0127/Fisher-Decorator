import ml_collections

from agents.fidec import FiDecAgent


class FiDecTraceNormV3Agent(FiDecAgent):
    """Backward-compatible alias of the finalized FiDec agent."""


def get_config():
    config = FiDecAgent.get_default_config()
    config.agent_name = 'fidec_trace_norm_v3'
    return config
