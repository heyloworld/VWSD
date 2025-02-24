# prompts.py
def get_prompt(query):
    return f"Can you caption this image with the '{query}' or not? Return Yes or No"

# Chain of Thought Prompt for Image-Query-Context Matching
def get_cot_prompt(query, context):

    prompt = f"""Input:
    - Query: {query}
    - Context: {context}
    Task: Determine if the provided image matches both the query and context.

    Please follow these steps to reach a Yes/No conclusion:

    1. Image Content Analysis
    - What are the main objects and elements visible in the image?
    - Analyze their:
        - Relative sizes and scale
        - Colors and visual appearance
        - Positioning and arrangement
        - Distinctive features
    - Note any text, labels, or markings
    - Describe the overall setting/environment

    2. Query Match Assessment for "{query}"
    - Is the specific {query} present in the image?
    - If visible, examine its:
        - Physical appearance and properties
        - Identifying characteristics
        - Current state or condition
        - Number/quantity if applicable
    - Compare against known traits of {query}
    - Document any identification uncertainties

    3. Context Match Assessment for "{context}"
    - Evaluate the connection to {context}
    - Look for context indicators:
        - Direct examples or instances
        - Related items or materials
        - Production or processing elements
        - Packaging or branding
        - Typical use cases
        - Associated equipment

    4. Combined Evaluation
    - {query} presence: (Met/Not Met)
    - {context} relationship: (Met/Not Met)
    - Connection analysis: How does {query} relate to {context} in this image?

    5. Confidence Assessment
    - Confidence in {query} identification: (High/Medium/Low)
    - Confidence in {context} connection: (High/Medium/Low)
    - Note uncertainty factors:
        - Image clarity and quality
        - Object visibility
        - Possible ambiguities
        - Alternative interpretations

    6. Final Decision
    - YES if:
        - {query} is clearly identifiable
        - Connection to {context} is evident
        - Confidence levels are adequate
    - NO if:
        - {query} or {context} requirements not met
        - Low confidence in identification
        - Significant uncertainty present

    Response Format:
    "Analyzing step by step:
    1. Image shows: [detailed description]
    2. Query ({query}) match: [analysis]
    3. Context ({context}) match: [analysis]
    4. Combined evaluation: [analysis]
    5. Confidence: [assessment]
    6. Therefore: [YES/NO] because [brief explanation]"
    """
    return prompt.strip()

def get_tot_prompt(query, context):
  
    prompt = f"""
    Input:
    - Query: {query}
    - Context: {context}
    Task: Explore multiple reasoning paths to determine if the image matches both query and context.

    Let's analyze through different perspectives:

    PATH A: Visual Feature Analysis
    1. Primary Visual Assessment
    - List all visible objects and elements
    - Document colors, shapes, sizes
    - Note spatial relationships
    - Initial impression: [Promising/Neutral/Unfavorable]

    2. Detailed Feature Matching for {query}
    Branch 1 - If clearly visible:
    - Examine defining characteristics
    - Compare with known {query} features
    - Confidence level: [High/Medium/Low]
    
    Branch 2 - If partially visible/ambiguous:
    - List visible features
    - Compare with potential alternatives
    - Confidence level: [High/Medium/Low]
    
    Branch 3 - If not visible:
    - Note absence
    - Check for related indicators
    - Confidence level: None

    3. {context} Integration
    - How does the identified element relate to {context}?
    - Rate contextual alignment: [Strong/Moderate/Weak]

    PATH B: Contextual Relationship Analysis
    1. {context} Environment Check
    Branch 1 - Direct Connection:
    - Look for immediate {context} indicators
    - Assess environment relevance
    - Confidence: [High/Medium/Low]
    
    Branch 2 - Indirect Connection:
    - Identify related elements
    - Evaluate circumstantial evidence
    - Confidence: [High/Medium/Low]
    
    Branch 3 - No Clear Connection:
    - Document absence of indicators
    - Consider alternative contexts
    - Confidence: None

    2. {query} Role Assessment
    - How does {query} typically relate to {context}?
    - Is this relationship evident in the image?
    - Confidence in relationship: [High/Medium/Low]

    PATH C: Uncertainty Analysis
    1. Image Quality Assessment
    - Resolution and clarity
    - Lighting conditions
    - Viewing angle
    - Impact on confidence: [High/Medium/Low]

    2. Ambiguity Evaluation
    - List possible alternative interpretations
    - Rate confidence in each interpretation
    - Impact on final decision: [Significant/Moderate/Minor]

    Decision Synthesis:
    1. Compile evidence from all paths:
    Path A conclusion: [Supporting/Neutral/Opposing]
    Path B conclusion: [Supporting/Neutral/Opposing]
    Path C impact: [Strengthens/Weakens] confidence

    2. Weighted Evaluation:
    - Strong evidence paths:
    - Weak evidence paths:
    - Conflicting evidence:

    3. Final Decision Logic:
    YES if:
    - Multiple paths support presence of {query}
    - Strong connection to {context} established
    - Uncertainty analysis supports confidence
    
    NO if:
    - Critical paths fail to support {query}
    - Weak or missing {context} connection
    - High uncertainty in key aspects

    Response Format:
    "Tree of Thought Analysis:
    1. Path A (Visual Features): [analysis]
    2. Path B (Context): [analysis]
    3. Path C (Uncertainty): [analysis]
    4. Evidence Synthesis: [summary]
    5. Therefore: [YES/NO] because [key evidence from multiple paths]"
    """
    return prompt.strip()