# main.py
import os

print("SMS Spam Detection Project")
print("1️⃣ Train Model")
print("2️⃣ Run Web App")

choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    os.system("python train_model.py")
elif choice == "2":
    os.system("streamlit run app.py")
else:
    print("Invalid choice! Please enter 1 or 2.")
