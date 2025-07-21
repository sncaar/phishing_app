import tkinter as tk
from tkinter import messagebox
import joblib
import xgboost as xgb
from features import extract_features


bst = xgb.Booster()
bst.load_model('xgb_bst_final.model')

scaler = joblib.load('scaler_final.joblib')

THRESHOLD = 0.60  # Olasılık eşik değeri

# === 2) Tkinter Arayüz Ayarları ===
root = tk.Tk()
root.title("URL Phishing Tespit Uygulaması")
root.geometry("540x200")
root.resizable(False, False)

label = tk.Label(root, text="URL Girin:", font=("Arial", 12))
label.pack(pady=(20, 5))

entry = tk.Entry(root, width=70, font=("Arial", 11))
entry.pack()

result_var = tk.StringVar()
result_var.set("Sonuç: Burada görünecek")
lbl_result = tk.Label(
    root,
    textvariable=result_var,
    font=("Arial", 14, "bold"),
    fg="black"
)
lbl_result.pack(pady=15)

# === 3) "Kontrol Et" Butonuna Bağlı Fonksiyon ===
def on_check():
    url = entry.get().strip()
    if not url:
        messagebox.showwarning("Uyarı", "Lütfen bir URL girin!")
        return

    try:
        # 1. Öznitelik çıkar
        df_vec = extract_features(url)          # DataFrame(1×87)
        # 2. Ölçekle
        x_sc = scaler.transform(df_vec)
        # 3. DMatrix'e çevirip modelden olasılık al
        dmat = xgb.DMatrix(x_sc)
        prob = bst.predict(dmat)[0]             # [0] → tek satır
        # 4. Etiketi belirle
        label = 'Phishing' if prob >= THRESHOLD else 'Legitimate'
        # 5. Sonucu göster ve renk kodlama yap (phishing kırmızı, legit yeşil)
        result_var.set(f"Sonuç: {label}  (Olasılık: {prob:.2f})")
        lbl_result.config(fg="red" if label == "Phishing" else "green")

    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu:\n{e}")

# === 4) Butonu Oluştur ve Konumlandır ===
btn = tk.Button(
    root,
    text="Kontrol Et",
    command=on_check,
    width=15,
    font=("Arial", 11)
)
btn.pack()

# === 5) Uygulamayı Başlat ===
root.mainloop()