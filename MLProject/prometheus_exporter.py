from prometheus_client import start_http_server, Counter, Gauge
import psutil

# 1. Inisialisasi port khusus untuk Prometheus Scraper
def start_exporter(port=8001):
    start_http_server(port)
    print(f"📊 Prometheus Exporter running at http://localhost:{port}/metrics")

# 2. Definisikan Metrik Eksplisit Sesuai Gambar 4 Acuan Dicoding
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total', 
    'Total jumlah HTTP request yang masuk ke API',
    ['method', 'endpoint', 'http_status']
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage',
    'Persentase penggunaan CPU sistem saat ini'
)

SYSTEM_RAM_USAGE = Gauge(
    'system_ram_usage',
    'Persentase penggunaan RAM sistem saat ini'
)

# Fungsi helper untuk memperbarui metrik hardware secara berkala
def update_system_metrics():
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
