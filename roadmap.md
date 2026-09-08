# Roadmap Hexkit

## Segment breakdown / heterogeneous treatment effects
After an experiment concludes, analysts often want to know: did the effect differ by device, channel, new vs. returning, geo? A tool that slices experiment results across pre-defined dimensions and flags statistically interesting subgroups — with appropriate warnings about multiple comparisons — would be a natural extension of the existing analysis tools.

## Novelty effect detector
A time-windowed view of treatment effect over the experiment's lifespan — does the lift decay after week 1? Is it a novelty bump or a sustained change? Pairs naturally with sequential analysis. Small scope: it's essentially a rolling-window effect size chart with a decay test.

## Make streamlit installable on local machines
Streamlit's uptime is never guaranteed. By making it installable on local machines, we bypass this dependency. Updates need to be pulled down manually to start with. We can look at fully packaging it in a later stage.
