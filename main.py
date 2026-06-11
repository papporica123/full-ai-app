
import io, refrom io import Bytes.io
import streamlit as streamlit
from huggingface_hub import InfrenceClient
import config
from groq import generate_response
Math_SYSTEM="""you are a Math mastermind.
solvee with clear steps and reasoning, correct notation, and a final answer.
Verify when possible; mention an alternative method briefly if relevant."""
CHAT_CSS="""
<style>
.wrap{max-height: 520px; overflow-y: auto; padding-right:P 6px;}
.card{border: 1px; solid #e6e6e6; background: #fff; border-radius: 10px; padding: 14px 16px; margin: 10px}