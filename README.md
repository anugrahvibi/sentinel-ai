
SentinelAl is a security-focused ML project that explores the detection of hidden backdoors in pre-trained Al models through behavioral analysis. Instead of relying on static checks or model metadata, the system treats models as black boxes and evaluates how their outputs change when exposed to controlled perturbations such as rare tokens or out-of-context inputs. By comparing baseline behavior against triggered behavior, the project demonstrates how compromised models can appear normal under standard testing yet fail silently when specific patterns are introduced. Built using Python, PyTorch, Hugging Face, FastAPI, and a lightweight dashboard, this project highlights real-world risks in the Al supply chain and emphasizes the importance of model verification before deployment.


## Results

Scan output on backdoor model:
- Verdict: BACKDOORED
- Risk score: 60%
- Label flips: 3/5 sentences
- Confidence jump on triggered inputs: ~99%