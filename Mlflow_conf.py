from mlflow.tracking import MlflowClient

client = MlflowClient()
registered_models = client.search_registered_models()

for rm in registered_models:
    print(f"Name: {rm.name}")
    for mv in rm.latest_versions:
        print(f" - Version: {mv.version}, Stage: {mv.current_stage}, Status: {mv.status}")


# mlflow server --backend-store-uri file:///F:/MLOPS/mlruns --default-artifact-root file:///F:/MLOPS/mlruns
