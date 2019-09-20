# Shakespeare-Generator
This little project generates Shakespeare plays by training its underlying n-gram word model with Shakespeare's entire corpus. Feel free to adjust what n is within the .ipynb file; just be warned that using n's over 11 will require some patience!

Note: for some reason, running the `shakespeare_working.ipynb` file yields no problems, but the exact same code ported over to `shakespeare.py` is having trouble. I'll be looking into possible fixes; for now, avoid `shakespeare.py`.

How to run Shakespeare Generator
1) Clone this repository via <a href="https://git-scm.com/"> git </a>.
2) If you don't have it already, <a href="https://pip.pypa.io/en/stable/installing/">download pip</a>
3) Open up terminal and navigate to the shakespeare-generator folder 
4) Type `pip install -r requirements.txt` inside Terminal
5) Then type `jupyter notebook`
6) Navigate to `shakespeare-working.ipynb` and enjoy!
