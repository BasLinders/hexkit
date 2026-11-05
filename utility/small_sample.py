"""
Shared settings for Boschloo's exact test in hexkit.

BOSCHLOO_MAX_PER_ARM is the largest arm that Small Sample Analysis runs
Boschloo on, and the size below which Experiment Analysis suggests it. It is
passed to BoschlooEngine explicitly instead of relying on foe's default (1,000),
because scipy's runtime grows steeply with sample size and one run blocks a
Streamlit worker: about 3s at 500 per arm versus about 20s at 1,000. Above this
limit the z-test is used, which is accurate at that size.
"""

BOSCHLOO_MAX_PER_ARM = 500
