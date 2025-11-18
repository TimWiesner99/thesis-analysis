# Future Analysis: Path Analysis of TiA Subscales

## Context
During discussion of moderation analysis (2025-11-18), we identified a potential future analysis examining relationships between TiA subscales.

## Key Points from Discussion

### Why Not Traditional Mediation?
- All TiA subscales measured simultaneously post-stimulus
- No temporal separation between variables
- Cannot establish clear causal ordering empirically
- Pre-existing traits (ATI, healthcare trust) are moderators, not mediators

### Why Path Analysis Could Still Be Valuable

**Theoretical Basis**: Strong theoretical foundation exists in literature for temporal/causal ordering of TiA subscales.

**Research Question**: Do the empirical relationships among TiA subscales confirm or challenge theoretical relationships established in previous literature?

**Proper Framing**:
- Call it "path analysis" or "structural equation modeling" rather than "mediation"
- Focus on relationships between constructs
- Be cautious with causal language
- Frame as testing theoretical model against data

**Potential Paths to Explore** (based on theoretical literature):
- Group → reliability (rc) → general trust (t)
- Group → understanding (up) → general trust (t)
- Group → familiarity (f) → general trust (t)
- Propensity (pro) as moderator of other paths?

### Considerations for Implementation

**Statistical Approach**:
- Use Structural Equation Modeling (SEM) with lavaan-style syntax
- Could use semopy or similar Python package
- Compare nested models
- Report fit indices (CFI, RMSEA, SRMR)

**Limitations to Acknowledge**:
- Cross-sectional data limits causal inference
- Model comparison can suggest "better fit" but not prove causality
- Need strong theoretical justification for any ordering

**Theoretical Literature to Reference**:
- Körber (2019) - original TiA scale development
- Any theoretical models about how different trust components develop/relate

## Next Steps (When Returning to This)
1. Review literature on theoretical ordering of TiA components
2. Specify a priori theoretical models to test
3. Implement SEM/path analysis in Python
4. Compare empirical findings to theoretical predictions
5. Discuss implications for trust construct validity

## Connection to Current Work
- Moderation analysis examines "for whom" uncertainty affects trust
- Path analysis would examine "through which components" trust operates
- Complementary but distinct research questions
