# Car Price Prediction - User Dataset

Files:
- car_data.csv : your dataset (must be saved in this project folder)
- Car_Price_Prediction_User.ipynb : Jupyter notebook with the pipeline
- train_model.py : script to retrain and save the best model
- car_price_model_user.joblib : trained model (created after running the notebook or train script)
- app_user.py : Streamlit app to interactively predict selling price
- requirements.txt : python packages required

Quick setup (Windows/macOS/Linux):

1. Create project folder and place all files there.
2. Create a virtual environment:
   - Windows PowerShell:
       python -m venv venv
       .\\venv\\Scripts\\Activate.ps1
   - macOS/Linux:
       python3 -m venv venv
       source venv/bin/activate

3. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

4. Run the training notebook:
   - Option A: jupyter notebook -> open Car_Price_Prediction_User.ipynb and run cells
   - Option B: python train_model.py

   After training, file `car_price_model_user.joblib` will be created.

5. Run the Streamlit app:
   streamlit run app_user.py

Notes:
- Ensure the dataset filename is `car_data.csv` and has the same columns as your original upload:
  Car_Name, Year, Selling_Price, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner
- The units of Present_Price / Selling_Price are preserved (e.g., lakhs). The model predicts in the same unit.
