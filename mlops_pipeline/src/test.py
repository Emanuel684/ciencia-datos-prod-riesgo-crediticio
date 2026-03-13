import pandas as pd

df = pd.read_excel("Base_de_datos.xlsx")

corr = df.corr(numeric_only=True)["Pago_atiempo"].sort_values(ascending=False)

print(corr)
