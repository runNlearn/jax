from jax import random

def config_tpu(tpu_name):
  from jax.config import config
  from cloud_tpu_client import Client
  c = Client(tpu_name)
  ip_info = c.network_endpoints()[0]
  ip = ip_info['ipAddress']
  port = ip_info['port']
  config.FLAGS.jax_xla_backend = "tpu_driver"
  config.FLAGS.jax_backend_target = f"grpc://{ip}:{port}"

config_tpu('moon')
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
