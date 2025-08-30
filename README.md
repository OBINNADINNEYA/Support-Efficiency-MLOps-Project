# Support Efficiency MLOps

End-to-end machine learning & causal inference pipeline on **31M GitHub issues**.  
We predict **issue complexity**, **routing (team assignment)**, and **resolution time**, and measure the **causal effect of automation (bots)** on efficiency.

## Components
- **Models**: Complexity, Routing, Resolution Time
- **Causal**: Propensity Score Matching & DiD
- **MLOps**: MLflow, Dockerized FastAPI, CI/CD
- **Dashboards**: Power BI efficiency trends

## Quickstart
```bash
make install
make train
make serve
