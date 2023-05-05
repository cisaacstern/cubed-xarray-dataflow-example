import os

import cubed
import cubed.random
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from cubed.runtime.executors.beam import BeamDagExecutor

work_dir = os.environ["WORK_DIR"]
job_name = os.environ["JOB_NAME"]
project = os.environ["PROJECT"]
region = os.environ["REGION"]
temp_location = os.environ["TEMP_LOCATION"]
staging_location = os.environ["STAGING_LOCATION"]
service_account_email = os.environ["SERVICE_ACCOUNT_EMAIL"]

spec = cubed.Spec(work_dir=work_dir, max_mem=2_000_000_000)

U = xr.DataArray(
    name='U',
    data=cubed.random.random((5, 1, 987, 1920), chunks=(1, 1, -1, -1), spec=spec),
    dims=["time", "face", "j", "i"],
)
V = xr.DataArray(
    name='V',
    data=cubed.random.random((5, 1, 987, 1920), chunks=(1, 1, -1, -1), spec=spec),
    dims=["time", "face", "j", "i"], 
)
dx = xr.DataArray(
    name='dx',
    data=cubed.random.random((1, 987, 1920), chunks=(1, -1, -1), spec=spec),
    dims=["face", "j", "i"],
)
dy = xr.DataArray(
    name='dy',
    data=cubed.random.random((1, 987, 1920), chunks=(1, -1, -1), spec=spec),
    dims=["face", "j", "i"],
)
ds = xr.merge([U, V, dx, dy])

omega = ds.U * ds.dx - ds.V * ds.dy
mean = omega.mean('time', skipna=False)  # skipna=False to avoid eager load, see xarray issue #7243

executor = BeamDagExecutor()

beam_options = PipelineOptions(
    runner="DataFlowRunner",
    job_name=job_name,
    project=project,
    region=region,
    temp_location=temp_location,
    staging_location=staging_location,
    use_public_ips=False,
    service_account_email=service_account_email,
    requirements_file="worker-requirements.txt",
    tmp_path=work_dir,
    # last two options may not be necessary
    save_main_session=True,
    pickle_library="cloudpickle",
)

m = mean.compute(executor=executor, options=beam_options)
print(m)
