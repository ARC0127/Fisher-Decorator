from agents.deflow import DeFlowAgent
from agents.deflowvf import DeFlowVFAgent
from agents.fidec import FiDecAgent
from agents.fidec_trace_norm_v3 import FiDecTraceNormV3Agent

agents = dict(
    deflow=DeFlowAgent,
    deflowvf=DeFlowVFAgent,
    fidec=FiDecAgent,
    fidec_trace_norm_v3=FiDecTraceNormV3Agent,
)
