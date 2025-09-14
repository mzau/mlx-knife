"""
Utilities for handling reasoning models and their output.

Ported from 1.x mlx_knife/reasoning_utils.py for 2.0 compatibility.

Different models use different formats for reasoning:
- MXFP4/GPT-OSS: <|channel|>analysis<|message|>REASONING<|end|>...<|channel|>final<|message|>ANSWER
- DeepSeek R1: <think>REASONING</think>ANSWER
- Claude: <thinking>REASONING</thinking>ANSWER
- QwQ: Similar to MXFP4
"""

import re
from typing import Dict, Optional, Tuple


class ReasoningExtractor:
    """Extract reasoning and final answer from model outputs."""
    
    # Model-specific patterns
    PATTERNS = {
        'gpt-oss': {
            'reasoning': r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
            'final': r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)',
            'markers': {
                'reasoning_start': '<|channel|>analysis<|message|>',
                'reasoning_end': '<|end|>',
                'final_marker': '<|channel|>final<|message|>',
                # Skip tokens that appear between reasoning and final
                'skip_tokens': ['<|start|>assistant<|channel|>final<|message|>', '<|start|>assistant', '<|start|>', '<|channel|>final<|message|>'],
                # Conditional skip tokens - only skip if at start of final section
                'conditional_skip': ['assistant']
            }
        },
        'deepseek': {
            'reasoning': r'<think>(.*?)</think>',
            'final': r'</think>(.*?)$',
            'markers': {
                'reasoning_start': '<think>',
                'reasoning_end': '</think>',
            }
        },
        'claude': {
            'reasoning': r'<thinking>(.*?)</thinking>',
            'final': r'</thinking>(.*?)$',
            'markers': {
                'reasoning_start': '<thinking>',
                'reasoning_end': '</thinking>',
            }
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> Optional[str]:
        """Detect reasoning model type from model name."""
        model_lower = model_name.lower()
        
        if 'gpt-oss' in model_lower:
            return 'gpt-oss'
        elif 'deepseek' in model_lower and 'r1' in model_lower:
            return 'deepseek'
        elif 'claude' in model_lower:
            return 'claude'
        elif 'qwq' in model_lower:
            return 'gpt-oss'  # QwQ uses similar format to GPT-OSS
        
        return None
    
    @classmethod
    def extract(cls, text: str, model_type: Optional[str] = None,
                model_name: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Extract reasoning and final answer from model output.
        
        Args:
            text: The full model output
            model_type: Explicit model type ('mxfp4', 'deepseek', etc.)
            model_name: Model name to auto-detect type
            
        Returns:
            Dictionary with 'reasoning', 'final_answer', and 'full_response'
        """
        # Auto-detect model type if not provided
        if not model_type and model_name:
            model_type = cls.detect_model_type(model_name)
        
        # If no model type detected, return text as-is
        if not model_type or model_type not in cls.PATTERNS:
            return {
                'reasoning': None,
                'final_answer': text,
                'full_response': text,
                'has_reasoning': False
            }
        
        patterns = cls.PATTERNS[model_type]
        
        # Extract reasoning
        reasoning_match = re.search(patterns['reasoning'], text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        # Extract final answer
        final_match = re.search(patterns['final'], text, re.DOTALL)
        final_answer = final_match.group(1).strip() if final_match else None
        
        # If no final answer found but we have reasoning,
        # the text after reasoning might be the answer
        if reasoning and not final_answer:
            # Try to find text after reasoning markers
            markers = patterns.get('markers', {})
            if 'reasoning_end' in markers:
                split_text = text.split(markers['reasoning_end'], 1)
                if len(split_text) > 1:
                    # Clean up any remaining markers
                    remaining = split_text[1]
                    for marker in markers.values():
                        remaining = remaining.replace(marker, '')
                    final_answer = remaining.strip()
        
        # If still no final answer, use full text minus reasoning markers
        if not final_answer:
            final_answer = text
            # Remove all known markers
            if model_type in cls.PATTERNS:
                markers = cls.PATTERNS[model_type].get('markers', {})
                for marker in markers.values():
                    final_answer = final_answer.replace(marker, '')
            final_answer = final_answer.strip()
        
        return {
            'reasoning': reasoning,
            'final_answer': final_answer,
            'full_response': text,
            'has_reasoning': bool(reasoning),
            'model_type': model_type
        }
    
    @classmethod
    def format_for_display(cls, extracted: Dict[str, Optional[str]],
                          show_reasoning: bool = False) -> str:
        """
        Format extracted content for display.
        
        Args:
            extracted: Output from extract()
            show_reasoning: Whether to include reasoning in output
            
        Returns:
            Formatted string for display
        """
        if not extracted.get('has_reasoning'):
            return extracted.get('final_answer', '')
        
        if show_reasoning:
            output = []
            if extracted.get('reasoning'):
                output.append("═══ Reasoning ═══")
                output.append(extracted['reasoning'])
                output.append("\n═══ Answer ═══")
            output.append(extracted.get('final_answer', ''))
            return '\n'.join(output)
        else:
            return extracted.get('final_answer', '')


class StreamingReasoningHandler:
    """Handle reasoning during streaming generation."""
    
    def __init__(self, model_type: Optional[str] = None):
        self.model_type = model_type
        self.buffer = ""
        self.reasoning_buffer = ""
        self.final_buffer = ""
        self.in_reasoning = False
        self.in_final = False
        self.markers = {}
        
        if model_type and model_type in ReasoningExtractor.PATTERNS:
            self.markers = ReasoningExtractor.PATTERNS[model_type].get('markers', {})
    
    def process_token(self, token: str) -> Tuple[str, bool]:
        """
        Process a streaming token.
        
        Args:
            token: The new token
            
        Returns:
            (output_token, should_display) - token to output and whether to display it
        """
        self.buffer += token
        
        # Check for reasoning start
        if not self.in_reasoning and self.markers.get('reasoning_start'):
            if self.markers['reasoning_start'] in self.buffer:
                self.in_reasoning = True
                self.reasoning_buffer = self.buffer.split(self.markers['reasoning_start'])[1]
                return ("", False)  # Don't display reasoning start marker
        
        # If in reasoning, buffer it
        if self.in_reasoning:
            self.reasoning_buffer += token
            
            # Check for reasoning end
            if self.markers.get('reasoning_end') and self.markers['reasoning_end'] in self.reasoning_buffer:
                self.in_reasoning = False
                self.in_final = True
                # Clean up reasoning buffer
                self.reasoning_buffer = self.reasoning_buffer.replace(self.markers['reasoning_end'], '')
                return ("", False)  # Don't display reasoning end marker
            
            return ("", False)  # Don't display reasoning content by default
        
        # If in final answer section
        if self.in_final:
            # Skip final answer markers
            if self.markers.get('final_marker') and self.markers['final_marker'] in token:
                return ("", False)
            
            self.final_buffer += token
            return (token, True)  # Display final answer
        
        # Default: display token if not in special section
        return (token, True)


class StreamingReasoningParser:
    """Parser for real-time streaming with reasoning model formatting."""
    
    def __init__(self, model_type: Optional[str] = None, hide_reasoning: bool = False):
        self.model_type = model_type
        self.hide_reasoning = hide_reasoning
        self.state = "WAITING"  # WAITING, IN_REASONING, IN_FINAL
        self.buffer = ""
        self.reasoning_content = ""
        self.patterns = {}
        
        if model_type and model_type in ReasoningExtractor.PATTERNS:
            self.patterns = ReasoningExtractor.PATTERNS[model_type].get('markers', {})
    
    def process_token(self, token: str):
        """
        Process a streaming token and yield formatted output.
        
        Args:
            token: New token from model
            
        Yields:
            Formatted output tokens for display
        """
        self.buffer += token
        
        # State: WAITING - looking for reasoning start
        if self.state == "WAITING":
            reasoning_start = self.patterns.get('reasoning_start')
            if reasoning_start and reasoning_start in self.buffer:
                # Found reasoning start
                before_reasoning = self.buffer.split(reasoning_start, 1)[0]
                
                # Yield any content before reasoning (but not control tokens)
                if before_reasoning.strip() and not before_reasoning.strip().startswith('<|'):
                    yield before_reasoning
                
                # Start reasoning section (only if not hiding reasoning)
                if not self.hide_reasoning:
                    yield "**[Reasoning]**\n\n"
                
                # Switch to reasoning state
                self.buffer = self.buffer.split(reasoning_start, 1)[1]
                self.state = "IN_REASONING"
                
                # Process remaining buffer recursively
                if self.buffer.strip():
                    yield from self.process_token("")
                return
            
            # Check if buffer might contain start of reasoning pattern
            if reasoning_start:
                # Check if buffer ends with partial pattern
                has_partial_match = False
                for i in range(1, min(len(reasoning_start) + 1, len(self.buffer) + 1)):
                    if self.buffer.endswith(reasoning_start[:i]):
                        has_partial_match = True
                        break
                
                if has_partial_match:
                    # Don't yield yet - might be building up to pattern
                    return
                
                # No partial match, safe to yield older content
                # Keep enough buffer to detect pattern
                pattern_len = len(reasoning_start)
                if len(self.buffer) > pattern_len:
                    to_yield = self.buffer[:-pattern_len]
                    self.buffer = self.buffer[-pattern_len:]
                    if to_yield:
                        yield to_yield
                    return
            
            # No reasoning pattern expected or very short buffer
            if not reasoning_start:
                yield token
        
        # State: IN_REASONING - collecting reasoning content
        elif self.state == "IN_REASONING":
            reasoning_end = self.patterns.get('reasoning_end')
            if reasoning_end and reasoning_end in self.buffer:
                # Found reasoning end
                reasoning_part = self.buffer.split(reasoning_end, 1)[0]
                
                # Yield reasoning content (only if not hiding reasoning)
                if reasoning_part and not self.hide_reasoning:
                    yield reasoning_part
                
                # Add separator (only if not hiding reasoning)
                if not self.hide_reasoning:
                    yield "\n\n---\n\n**[Answer]**\n\n"
                
                # Switch to final state
                self.buffer = self.buffer.split(reasoning_end, 1)[1]
                self.state = "IN_FINAL"
                self._final_content_started = False  # Track if we've started outputting final content
                
                # Skip intermediate control tokens
                skip_tokens = self.patterns.get('skip_tokens', [])
                for skip_token in skip_tokens:
                    self.buffer = self.buffer.replace(skip_token, '')
                
                # Skip final marker when we find it
                final_marker = self.patterns.get('final_marker')
                if final_marker and final_marker in self.buffer:
                    self.buffer = self.buffer.split(final_marker, 1)[1]
                
                # Process remaining buffer
                if self.buffer.strip():
                    yield from self.process_token("")
                return
            
            # Still in reasoning, yield the content (only if not hiding reasoning)
            if not self.hide_reasoning:
                yield token
        
        # State: IN_FINAL - normal streaming of final answer
        elif self.state == "IN_FINAL":
            # Check for control tokens from patterns that should be filtered
            skip_tokens = self.patterns.get('skip_tokens', [])
            conditional_skip = self.patterns.get('conditional_skip', [])
            
            # Check if buffer contains any skip tokens and filter them out
            for skip_token in skip_tokens:
                if skip_token in self.buffer:
                    # Remove the skip token and continue
                    self.buffer = self.buffer.replace(skip_token, '')
                    # Process remaining buffer if any
                    if self.buffer.strip():
                        yield from self.process_token("")
                    return
            
            # Check for final marker and filter it too
            final_marker = self.patterns.get('final_marker')
            if final_marker and final_marker in self.buffer:
                # Split at final marker and yield only content after it
                parts = self.buffer.split(final_marker, 1)
                if len(parts) > 1:
                    self.buffer = parts[1]
                    if self.buffer.strip():
                        yield from self.process_token("")
                    return
                else:
                    # Just the marker itself, skip it
                    return
            
            # Check conditional skip tokens - only at start of final section
            if not getattr(self, '_final_content_started', False):
                for cond_token in conditional_skip:
                    if token.strip() == cond_token:
                        # Skip this token at the beginning of final section
                        return
                # Mark that final content has started after first non-conditional token
                if token.strip() and not any(token.strip() == ct for ct in conditional_skip):
                    self._final_content_started = True
            
            # Check if we might be building up to a skip token - be conservative
            potential_skip = False
            for skip_token in skip_tokens:
                if skip_token.startswith(token) or any(skip_token.startswith(self.buffer[-i:]) for i in range(1, min(len(skip_token), len(self.buffer)) + 1)):
                    potential_skip = True
                    break
            
            if potential_skip:
                # Don't yield yet, might be building up to a skip token
                return
            
            # Normal token in final answer - safe to yield
            yield token
    
    def finalize(self):
        """
        Finalize parsing and yield any remaining buffer content.
        Call this when streaming is complete.
        """
        if self.buffer.strip():
            if self.state == "WAITING":
                # No reasoning was found, output as normal text
                yield self.buffer
            elif self.state == "IN_REASONING":
                # Reasoning never ended, output what we have
                yield self.buffer
            elif self.state == "IN_FINAL":
                # Final answer content
                yield self.buffer