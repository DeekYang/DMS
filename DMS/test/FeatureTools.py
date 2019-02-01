#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
import featuretools as ft

if __name__ == "__main__":

	data = ft.demo.load_mock_customer()
	transactions_df  = data["transactions"].merge(data["sessions"]).merge(data["customers"])
	products_df = data["products"]

	es = ft.EntitySet()
	
	es = es.entity_from_dataframe(entity_id="transactions",dataframe=transactions_df,index="transaction_id",variable_types={"product_id": ft.variable_types.Categorical})
	es = es.entity_from_dataframe(entity_id="products",dataframe=products_df,index="product_id")
	
	new_relationship = ft.Relationship(es["products"]["product_id"],es["transactions"]["product_id"])
	es = es.add_relationship(new_relationship)
	
	print(es['transactions'].variables)

	feature_matrix, feature_defs = ft.dfs(entityset=es,target_entity="products",agg_primitives=["count",'sum'],)
	print(feature_matrix)
	print(feature_defs)