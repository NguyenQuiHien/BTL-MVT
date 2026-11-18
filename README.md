# Hướng dẫn chạy chương trình mô phỏng của nhóm

Chương trình dùng Optical Networking Gym để so sánh cách KSP-FF và LS-BM-KSP cấp tuyến, điều chế và phổ cho từng yêu cầu. Bạn có thể chạy một episode để kiểm tra agent, mở demo Streamlit để xem từng bước, hoặc quét nhiều mức tải và công suất để lấy số liệu.

## 1. Chuẩn bị và cài đặt
Trong PowerShell, dùng Python của môi trường ảo hiện có để cài thêm các thư viện cho demo và chạy kiểm thử tích hợp.

```powershell
cd D:\BTL-MVT\optical-networking-gym
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research,demo]"
.\.venv\Scripts\python.exe -m pytest -q tests/team
```

**Trên máy mới.** Cài Git, Python 3.12 và Visual Studio Build Tools với thành phần Desktop development with C++.

```powershell
git clone https://github.com/quangnguyenthanhquang-tech/BTL-MVT.git
cd BTL-MVT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research,demo]"
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe -m pytest -q tests/team
```

Trong VS Code, dùng **Python: Select Interpreter** và chọn `.venv\Scripts\python.exe`. Các lệnh bên dưới đều chạy từ thư mục chứa `pyproject.toml`.

## 2. Cấu trúc chương trình

| Đường dẫn | Vai trò |
|---|---|
| team/config.py | Ba kịch bản và danh mục topology dùng chung |
| team/env_factory.py | Tạo môi trường cho CLI và demo |
| team/agents/rmsa_agents.py | KSPFFAgent và LSBMKSPAgent, chọn và kiểm tra action |
| team/runner.py | Vòng reset–agent–step và chỉ số cuối episode |
| team/main.py | Chọn agent/cấu hình qua CLI, lưu JSON |
| team/experiments.py | Quét tham số và xuất CSV/JSON |
| demo/app.py | Giao diện Streamlit |
| demo/sim_core.py | Thu log, chụp trạng thái, tổng hợp thí nghiệm |
| demo/csv_loader.py | Đọc và kiểm tra dữ liệu CSV |
| data/member4/ | Dữ liệu gốc được giữ nguyên để đưa lên Git |


Luồng chạy một episode bắt đầu từ `team/main.py`. File này tạo môi trường và agent rồi giao cho `runner.py` điều khiển từng bước. Demo sử dụng cùng factory và hai agent, đồng thời thu thêm dữ liệu từng yêu cầu để vẽ biểu đồ.

## 3. Chạy chương trình nhóm

### 3.1 Chạy một episode

Chạy từ thư mục có `pyproject.toml`, `team/` và `demo/`:

```powershell
.\.venv\Scripts\python.exe -m team.main --algorithm ksp_ff --seed 42 --requests 1000 --load 300
.\.venv\Scripts\python.exe -m team.main --algorithm ls_bm_ksp --seed 42 --requests 1000 --load 300
.\.venv\Scripts\python.exe -m team.env_test
```

Thêm `--scenario 2` hoặc `--scenario 3` để chọn kịch bản. `--power 0` đặt công suất 0 dBm, bỏ tham số thì dùng mặc định của ONG. `python -m team.main` vẫn chạy First-Fit với 100 yêu cầu như phiên bản cũ. JSON mới lưu cả cấu hình hiệu lực và phiên bản. `total_reward` không phải số yêu cầu được chấp nhận.

### 3.2 Mở demo Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

Mở URL Streamlit in trong terminal. Chọn topology, kịch bản và tham số trên thanh bên. Tab mô phỏng trực tiếp hiển thị hai thuật toán. Tab thí nghiệm hỗ trợ quét tải hoặc công suất và tải log, bảng theo seed, bảng tổng hợp cùng JSON cấu hình.

Tab CSV: nhập `data/member4/kb1` (hoặc kb2, kb3), hoặc tải file lên. Đường dẫn tương đối được tính từ thư mục chạy lệnh. Có thể chọn bảng tổng hợp đã gửi bằng tiêu đề tiếng Việt; chương trình giữ độ lệch chuẩn có sẵn.

### 3.3 Quét tải hoặc công suất

```powershell
.\.venv\Scripts\python.exe -m team.experiments --scenario 1 --axis load --values 50 100 150 200 250 300 --seeds 42 43 44 --requests 1000
.\.venv\Scripts\python.exe -m team.experiments --scenario 1 --axis power --values -8 -6 -4 -2 0 2 4 6 8 --load 300 --seeds 42 43 44 --requests 1000
```

Mỗi lệnh tạo thư mục mới trong `team/results/experiments/`, gồm `requests.csv`, `runs.csv`, `summary.csv`, `config.json`. `--warmup 0.1` bỏ 10% yêu cầu đầu khi tổng hợp; log gốc vẫn giữ toàn bộ. Quét công suất là chức năng đã tích hợp, không có nghĩa bộ thí nghiệm công suất đầy đủ đã được chạy.

## 4. Cấu hình mô phỏng

Chọn cấu hình bằng tham số dòng lệnh khi chạy `team.main`. Ví dụ, `--scenario 2 --load 300 --seed 42` chạy kịch bản siết phổ tại 300 Erlang với seed 42.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--algorithm` | `first_fit` | Chọn `first_fit`, `ksp_ff` hoặc `ls_bm_ksp` |
| `--topology` | `nobel-eu` | Chọn mạng Châu Âu hoặc `ring_4` |
| `--scenario` | 1 | Chọn bộ số slot và bit rate trong bảng dưới |
| `--load` | 300 | Tải chào, đơn vị Erlang |
| `--seed` | 42 | Hạt giống sinh lưu lượng |
| `--requests` | 100 | Số yêu cầu trong một episode |
| `--power` | Theo topology preset | Công suất phát theo dBm. Giá trị 0 đặt công suất ở 0 dBm |

| Kịch bản | Slot trên mỗi link | Bit rate yêu cầu (Gb/s) |
|---|---:|---|
| KB1 – Mạng chuẩn | 320 | 10, 40, 100, 400 |
| KB2 – Siết phổ | 80 | 10, 40, 100, 400 |
| KB3 – Đa dịch vụ | 320 | 40, 100, 400 |

Môi trường xét tối đa 3 tuyến ứng viên và 4 định dạng điều chế cho mỗi yêu cầu. Khi so sánh hai agent, dùng cùng topology, kịch bản, tải, công suất, seed và số yêu cầu.

## 5. Giao diện thuật toán

Mỗi agent cung cấp tên và phương thức:

```python
class CustomAgent:
    name = "custom_agent"

    def solution(self, obs, info, env):
        # Chọn và trả về action theo giao diện của môi trường.
        raise NotImplementedError
```

| Đầu vào | Nội dung |
|---|---|
| `obs` | Quan sát hiện tại |
| `info` | Thông tin bổ sung từ `reset()` hoặc `step()` |
| `env` | Môi trường, dùng khi cần lấy action mask hoặc ngữ cảnh thuật toán |

First-Fit hiện lấy mask từ `info.get("mask")`; nếu không có thì gọi `env.action_masks()`, sau đó dùng `select_first_fit_action(mask)`.


Hai lớp `KSPFFAgent` và `LSBMKSPAgent` trong `team/agents/rmsa_agents.py` gọi lần lượt các policy `jocn_ksp_ff_bm` và `jocn_ls_bm_ksp` của ONG. Trước khi trả action, chúng kiểm tra chỉ số có thuộc action_space và được mask cho phép. Cả dòng lệnh và demo sử dụng hai lớp này.

## 6. Vòng lặp môi trường

```python
obs, info = env.reset(seed=seed)
action = agent.solution(obs, info, env)
obs, reward, terminated, truncated, info = env.step(action)
```

`reset()` khởi tạo episode. `step()` thực hiện quyết định và trả về quan sát mới, reward, hai cờ kết thúc và thông tin bổ sung. Runner dừng khi `terminated or truncated` đúng. `main.py` gọi `env.close()` trong `finally` để đóng môi trường cả khi xảy ra lỗi.

Demo chụp ma trận phổ ngay sau khi áp dụng action và trước khi tiến thời gian sang yêu cầu tiếp theo. Điều này giúp ghép tuyến vừa chọn với đúng trạng thái phổ trên màn hình.

## 7. Kết quả JSON và CSV

Kết quả nằm trong `team/results/`, với tên dạng `first_fit_<thời gian UTC>.json`. Hậu tố `Z` biểu thị UTC; giờ Việt Nam bằng UTC cộng 7 giờ. Mỗi lần chạy tạo một file mới.

Ví dụ rút gọn từ lần kiểm thử First-Fit của nhóm (chỉ trích một số trường):

```json
{
  "algorithm": "first_fit",
  "config": {
    "seed": 42,
    "load": 300.0,
    "episode_length": 100
  },
  "metrics": {
    "steps": 100,
    "total_reward": 96.27289358129727,
    "request_blocking_ratio": 0.01,
    "bit_rate_blocking_ratio": 0.03289473684210526
  }
}
```

| Trường | Ý nghĩa |
|---|---|
| `algorithm` | Tên agent |
| `config` | Cấu hình lần chạy được xuất |
| `steps` | Số bước thực hiện |
| `total_reward` | Tổng reward trong episode |
| `request_blocking_ratio` | Tỷ lệ yêu cầu bị chặn, lấy từ `episode_service_blocking_rate` |
| `bit_rate_blocking_ratio` | Tỷ lệ bit rate bị chặn, lấy từ `episode_bit_rate_blocking_rate` |

Các tỷ lệ được lưu dưới dạng số từ 0 đến 1; nhân 100 để đổi sang phần trăm. Ví dụ trên có tỷ lệ chặn yêu cầu 1% và tỷ lệ chặn bit rate khoảng 3,2895%. Hai chỉ số có thể khác nhau vì các yêu cầu có bit rate khác nhau.

Reward phụ thuộc cách môi trường định nghĩa hàm thưởng; không diễn giải tổng reward trực tiếp thành tỷ lệ cấp phát thành công.

JSON của bản tích hợp còn có `scenario`, `effective_config` (tham số thực tế của môi trường) và `provenance` (phiên bản phần mềm, thông tin Git).

| File của một lần quét bằng team.experiments | Cách đọc |
|---|---|
| requests.csv | Mỗi dòng là một yêu cầu. Xem bit rate, kết quả chấp nhận, điều chế và GSNR |
| runs.csv | Mỗi dòng ứng với một thuật toán, một mức quét và một seed |
| summary.csv | Trung bình và độ lệch chuẩn giữa các seed ở từng mức quét |
| config.json | Cấu hình hiệu lực của các lần chạy, biến quét và tỷ lệ warm-up |

Với dữ liệu cũ trong `data/member4/`, mức sử dụng phổ được lấy sau khi môi trường có thể đã tiến thời gian. Bản tích hợp lấy ngay sau hành động. Khi so sánh biểu đồ, ghi rõ cách lấy mẫu. Thời gian mới gồm chọn và kiểm tra action, thực hiện bước môi trường và chụp phổ.

## 8. Lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| `No module named team` | Chạy từ thư mục gốc repository; kiểm tra `team/main.py` và các file `__init__.py` |
| `No module named optical_networking_gym` | Dùng đúng `.venv` và chạy lại lệnh cài editable trong mục 1 |
| Lỗi liên quan module Cython | Kiểm tra compiler và build lại bằng `setup.py build_ext --inplace` trong đúng môi trường |
| Không thấy JSON | Kiểm tra `team/results/` và đường dẫn chương trình in ra; nếu có traceback, xử lý lỗi trước |
| Tỷ lệ chặn lớn hơn 0 | Đây có thể là kết quả mô phỏng bình thường; kiểm tra cấu hình và thuật toán để phân tích |
| Báo thiếu Streamlit | Cài extra `demo` bằng lệnh ở mục 1 và dùng Python trong cùng .venv để mở giao diện |
| Tab CSV không tìm thấy thư mục | Mở Streamlit từ thư mục gốc repo và nhập đầy đủ `data/member4/kb1`, `data/member4/kb2` hoặc `data/member4/kb3` |

## 9. Ghi nhận phiên bản và quản lý mã nguồn

Ghi nhận commit và các gói đang cài để hỗ trợ tái lập:

```powershell
git rev-parse HEAD
.\.venv\Scripts\python.exe -m pip freeze > requirements-team.txt
```

`requirements-team.txt` là bản chụp môi trường hiện tại; cần kiểm tra các dòng đường dẫn cục bộ hoặc editable trước khi dùng trên máy khác.

Mỗi thành viên làm trên nhánh riêng. Không đưa `.venv`, cache và file build vào Git. Thống nhất cách lưu kết quả thực nghiệm với nhóm; giữ nguyên giấy phép và ghi nhận nguồn Optical Networking Gym.