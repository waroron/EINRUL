# ベースイメージとしてosrf/ros2:develを使用
FROM tiryoh/ros2-desktop-vnc:humble

# Python3とpipを最新版に更新
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python3のバージョンを確認
RUN python3 --version

# 任意の追加コマンド
# 例: ROS2パッケージや依存関係のインストール
# RUN apt-get update && apt-get install -y <your-dependencies>

# コンテナ起動時に実行されるコマンド
# CMD ["bash"]