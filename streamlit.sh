pip install streamlit
pip install openai
pip install python-dotenv
pip install chromadb
pip install pdfplumber

if ! grep -q "deb http://ftp.debian.org/debian stable main" /etc/apt/sources.list; then
  echo "deb http://ftp.debian.org/debian stable main" >> /etc/apt/sources.list
fi
apt update && apt install -y sqlite3

cd /home/site/wwwroot
python -m streamlit run chat_interface.py --server.port 8000
