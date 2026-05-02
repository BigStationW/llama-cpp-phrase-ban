# Use case
This repository allows you to ban phrases on llama.cpp.

<img width="720" alt="DISPLAY" src="https://github.com/user-attachments/assets/2d57f4e0-f664-49ec-9e47-fa7702232b58" />

# Installation

1) [Open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```
git clone https://github.com/BigStationW/llama-cpp-phrase-ban
```

# Usage

1) **Launch the llama.cpp Server**

By default, it runs on port 8080, unless you specified a different port using the ``--port`` flag.
 
2) **Configure Ports**

Open [launch.bat](https://github.com/BigStationW/llama-cpp-phrase-ban/blob/fe85804be5bec94d9ba97e23fc431ce333bf76a8/launch.bat#L6) in Notepad so you can set both the llama.cpp server port and the proxy port.

*(For example if you set ``LLAMA_PORT=8080`` and ``PROXY_PORT=5001``, the Ui will be accessible, at http://127.0.0.1:5001)*

3) **Start the Proxy**

Double click on ``launch.bat`` and navigate to your configured proxy address *(e.g. http://127.0.0.1:5001)*

4) **Ban Phrases**

To ban phrases you simply type them on ``banned_phrases.txt``. Once the file is saved, the changes are automatically applied.

5) **Test It Out**

You're good to go. Send a message through the UI and verify that banned phrases are being filtered correctly.
