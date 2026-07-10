Nível 4: Engenharia de Recursos (Criando Novas Colunas)

'    Extraindo Datas: Assumindo que a coluna Data_Pedido já está no formato datetime, crie duas novas colunas: Ano e Mes, extraídas diretamente da data do pedido.'

'    Cálculo de Custo: Você tem o Valor_Venda e o Lucro. Crie uma nova coluna chamada Custo subtraindo o lucro do valor de venda.'
'
    Sinalizador de Rentabilidade: Crie uma nova coluna booleana chamada Alta_Rentabilidade que retorna True se o Lucro for maior que 200.00, e False caso contrário.
'
Nível 5: Filtragem e Classificação Avançadas

'    Melhores Resultados: Encontre os 5 principais pedidos com o maior Lucro absoluto em todo o conjunto de dados.'

'    Margens Negativas: Filtre o DataFrame para encontrar todos os pedidos onde o Lucro é negativo (menor que 0), significando que a empresa perdeu dinheiro naquela venda.'
"
    Filtragem Complexa: Encontre todos os pedidos do segmento Consumidor na região Sudeste onde a Categoria é exatamente "Vestuário"."

Nível 6: Tabelas Dinâmicas e Agregação Avançada

'    Mágica da Tabela Dinâmica (Pivot Table): Crie uma Tabela Dinâmica onde as linhas são a Categoria, as colunas são a Regiao e os valores são a soma da Quantidade vendida. Preencha quaisquer valores NaN com 0.
'

'    Melhor Cliente: Agrupe os dados por ID_Cliente e calcule a soma de Valor_Venda. Classifique o resultado para encontrar o único ID_Cliente que gastou mais dinheiro no total.
'

    Produtos Únicos: Descubra exatamente quantos produtos únicos (Produto) estão listados no conjunto de dados.