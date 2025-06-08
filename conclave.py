import matplotlib.pyplot as plt
import random
import copy
from datetime import datetime, timedelta

# Existing Voo, Pessoa, Cromossomo classes (assuming they are already defined correctly from previous turns)
class Voo:
    """Representa um voo com origem, destino, horários e preço."""
    def __init__(self, origem, destino, horario_saida, horario_chegada, preco):
        self.origem = origem
        self.destino = destino
        self.horario_saida = horario_saida
        self.horario_chegada = horario_chegada
        self.preco = preco
    
    def __str__(self):
        # Formata o horário para HH:MM, útil para exibição
        return f"{self.origem}->{self.destino}: {self.horario_saida.strftime('%H:%M')}-{self.horario_chegada.strftime('%H:%M')} (€{self.preco:.0f})"

class Pessoa:
    """
    Representa uma pessoa com seu nome, cidade de origem, e as listas
    de voos de IDA e VOLTA que estão DISPONÍVEIS para ela.
    """
    def __init__(self, nome, origem_cidade, voos_ida_disponiveis, voos_volta_disponiveis):
        self.nome = nome # Ex: "Pessoa_MAD"
        self.origem_cidade = origem_cidade # Ex: "MAD"
        self.voos_ida = voos_ida_disponiveis # Lista de objetos Voo de MAD para FCO
        self.voos_volta = voos_volta_disponiveis # Lista de objetos Voo de FCO para MAD

    def __str__(self):
        return f"{self.nome} (de {self.origem_cidade})"

class Cromossomo:
    """
    Representa uma solução completa para todas as pessoas, ou seja,
    uma seleção específica de um voo de ida e um voo de volta para cada pessoa.
    """
    def __init__(self, voos_ida_selecionados, voos_volta_selecionados):
        # Cada item na lista é um objeto Voo (o voo ESCOLHIDO para aquela pessoa)
        self.voos_ida = voos_ida_selecionados 
        self.voos_volta = voos_volta_selecionados
        self.fitness = 0 # O valor de aptidão da solução

class AlgoritmoGenetico:
    def __init__(self, pessoas, destino='FCO', tamanho_populacao=100, taxa_mutacao=0.1, 
                 taxa_elite=0.05, geracoes_sem_melhoria_max=50, 
                 peso_custo=1.0, peso_gap_chegada=100.0, peso_gap_saida=100.0):
        self.pessoas = pessoas # Lista de objetos Pessoa
        self.destino = destino # Aeroporto de encontro (Ex: 'FCO')
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_elite = taxa_elite
        self.geracoes_sem_melhoria_max = geracoes_sem_melhoria_max
        
        self.populacao = []
        self.melhor_fitness = float('-inf') # Inicializa com valor bem baixo para maximização
        self.geracoes_sem_melhoria = 0

        # Novos atributos para os pesos do fitness
        self.peso_custo = peso_custo
        self.peso_gap_chegada = peso_gap_chegada
        self.peso_gap_saida = peso_gap_saida
        
    def carregar_voos_arquivo(self, arquivo_path):
        """
        Carrega voos do arquivo .txt e os agrupa por cidade para facilitar a criação de Pessoas.
        Formato esperado: origem, destino, horario_saida, horario_chegada, preco
        Exemplo: MAD, FCO, 11:01, 12:39, 260
        """
        # voos_por_cidade armazena: {'CIDADE_X': {'ida': [Voo1, Voo2], 'volta': [Voo3, Voo4]}}
        voos_por_cidade = {} 
        
        try:
            with open(arquivo_path, 'r', encoding='utf-8') as arquivo:
                for linha in arquivo:
                    linha = linha.strip()
                    if not linha or linha.startswith('#'):  # Ignora linhas vazias e comentários
                        continue
                    
                    partes = [parte.strip() for parte in linha.split(',')]
                    if len(partes) != 5:
                        print(f"Aviso: Linha ignorada (formato incorreto): {linha}")
                        continue
                    
                    origem_voo_str, destino_voo_str, hora_saida_str, hora_chegada_str, preco_str = partes
                    
                    try:
                        # Usamos uma data base fixa para parsear apenas horários, já que a data não importa para a duração
                        data_base = datetime(2024, 6, 15) 
                        
                        horario_saida = data_base.replace(
                            hour=int(hora_saida_str.split(':')[0]), 
                            minute=int(hora_saida_str.split(':')[1])
                        )
                        horario_chegada = data_base.replace(
                            hour=int(hora_chegada_str.split(':')[0]), 
                            minute=int(hora_chegada_str.split(':')[1])
                        )
                        
                        # Se o voo chega no dia seguinte, ajustamos a data de chegada
                        if horario_chegada <= horario_saida:
                            horario_chegada += timedelta(days=1)
                        
                        preco = float(preco_str)
                        
                        voo = Voo(origem_voo_str, destino_voo_str, horario_saida, horario_chegada, preco)
                        
                        # Inicializa as entradas no dicionário se ainda não existirem
                        voos_por_cidade.setdefault(voo.origem, {'ida': [], 'volta': []})
                        voos_por_cidade.setdefault(voo.destino, {'ida': [], 'volta': []})

                        # Classifica o voo como "ida" ou "volta" em relação ao destino principal (self.destino)
                        if voo.destino == self.destino: # Voo indo para FCO (ida da pessoa)
                            voos_por_cidade[voo.origem]['ida'].append(voo)
                        elif voo.origem == self.destino: # Voo saindo de FCO (volta da pessoa para sua origem)
                            # O destino do voo de volta é a cidade de origem da pessoa
                            voos_por_cidade[voo.destino]['volta'].append(voo) 
                    
                    except (ValueError, IndexError) as e:
                        print(f"Erro ao processar linha: {linha} - {e}")
                        continue
        
        except FileNotFoundError:
            print(f"Arquivo '{arquivo_path}' não encontrado. Gerando dados de exemplo para demonstração.")
            return self.gerar_dados_exemplo()
        
        return voos_por_cidade

    def criar_pessoas_from_arquivo(self, arquivo_path):
        """
        Cria objetos Pessoa a partir dos dados de voos carregados.
        Cada Pessoa é criada para uma cidade de origem específica com seus voos de ida e volta disponíveis.
        """
        voos_agrupados_por_cidade = self.carregar_voos_arquivo(arquivo_path)
        pessoas = []
        
        print("\n--- Voos carregados e agrupados por cidade ---")
        # Itera sobre as cidades que atuam como origem para os passageiros
        for cidade_origem, voos_dict in voos_agrupados_por_cidade.items():
            # Uma pessoa não "origina" do próprio aeroporto de destino do encontro
            if cidade_origem != self.destino: 
                ida_count = len(voos_dict['ida'])
                volta_count = len(voos_dict['volta'])
                print(f"  {cidade_origem}: {ida_count} voos de ida, {volta_count} voos de volta")
                
                # Só cria uma Pessoa se houver voos de ida E volta disponíveis para ela
                if voos_dict['ida'] and voos_dict['volta']:
                    nome_pessoa = f"Pessoa_{cidade_origem}"
                    pessoa = Pessoa(nome_pessoa, cidade_origem, voos_dict['ida'], voos_dict['volta']) 
                    pessoas.append(pessoa)
                else:
                    print(f"    AVISO: {cidade_origem} não tem voos de ida e/ou volta suficientes. Esta cidade não será incluída.")
        
        return pessoas
        
    def gerar_populacao_inicial(self):
        """
        Gera a população inicial de cromossomos aleatoriamente.
        Cada cromossomo é uma combinação de voos escolhidos aleatoriamente
        para cada pessoa.
        """
        if not self.pessoas:
            raise ValueError("Não há pessoas registradas para gerar a população. Carregue os dados primeiro.")

        self.populacao = []
        for _ in range(self.tamanho_populacao):
            voos_ida_selecionados = []
            voos_volta_selecionados = []
            
            for pessoa in self.pessoas:
                if not pessoa.voos_ida:
                    raise ValueError(f"Pessoa {pessoa.nome} não tem voos de ida disponíveis para seleção inicial.")
                if not pessoa.voos_volta:
                    raise ValueError(f"Pessoa {pessoa.nome} não tem voos de volta disponíveis para seleção inicial.")
                    
                voos_ida_selecionados.append(random.choice(pessoa.voos_ida))
                voos_volta_selecionados.append(random.choice(pessoa.voos_volta))
            
            cromossomo = Cromossomo(voos_ida_selecionados, voos_volta_selecionados)
            self.populacao.append(cromossomo)
    
    def calcular_fitness(self, cromossomo):
        """
        Calcula o fitness de um cromossomo (solução).
        O fitness é baseado em:
        1. Custo total dos voos (minimizar)
        2. Tempo de espera na van para chegada (minimizar o gap entre o primeiro e o último a chegar)
        3. Tempo de espera para a van de partida (minimizar o gap entre o primeiro e o último a sair)
        
        Um fitness mais alto (menos negativo) indica uma solução melhor.
        """
        # Verificações básicas para evitar erros em cromossomos incompletos (embora não devesse ocorrer com a geração atual)
        if not cromossomo.voos_ida or not cromossomo.voos_volta:
            return -1_000_000_000 # Penalidade muito alta

        # 1. Custo total dos voos
        custo_total = sum(voo.preco for voo in cromossomo.voos_ida + cromossomo.voos_volta)
        
        # 2. Tempo de espera para o encontro (voos de ida)
        horarios_chegada_ida = [voo.horario_chegada for voo in cromossomo.voos_ida]
        primeiro_chegada_ida = min(horarios_chegada_ida)
        ultimo_chegada_ida = max(horarios_chegada_ida)
        # O "gap" de chegada é a diferença entre o último e o primeiro a chegar
        gap_chegada_ida = (ultimo_chegada_ida - primeiro_chegada_ida).total_seconds() / 3600 # em horas
        
        # 3. Tempo de espera para a partida (voos de volta)
        horarios_saida_volta = [voo.horario_saida for voo in cromossomo.voos_volta]
        primeiro_saida_volta = min(horarios_saida_volta)
        ultimo_saida_volta = max(horarios_saida_volta)
        # O "gap" de saída é a diferença entre o último e o primeiro a sair
        gap_saida_volta = (ultimo_saida_volta - primeiro_saida_volta).total_seconds() / 3600 # em horas
        
        # Pesos para cada componente do fitness. Valores negativos porque queremos minimizar.
        # Ajuste esses pesos para dar mais importância ao custo ou ao tempo de espera.
        # Agora usando os pesos definidos na inicialização da classe
        fitness = -(
            self.peso_custo * custo_total + 
            self.peso_gap_chegada * gap_chegada_ida +
            self.peso_gap_saida * gap_saida_volta
        )
        
        return fitness
    
    def avaliar_populacao(self):
        """Calcula e atribui o fitness a todos os cromossomos na população atual."""
        for cromossomo in self.populacao:
            cromossomo.fitness = self.calcular_fitness(cromossomo)
    
    def selecao_torneio(self, tamanho_torneio=3, taxa_nao_melhor=0.05):
        """
        Seleciona um pai usando o método de torneio.
        Competidores são escolhidos aleatoriamente, e o melhor é selecionado,
        com uma pequena chance de não escolher o melhor para manter a diversidade.
        """
        if len(self.populacao) < 2: # Precisa de pelo menos 2 para torneio decente
            return self.populacao[0] if self.populacao else None

        # Garante que o tamanho do torneio não exceda a população
        tamanho_torneio = min(tamanho_torneio, len(self.populacao))
        
        competidores = random.sample(self.populacao, tamanho_torneio)
        competidores.sort(key=lambda x: x.fitness, reverse=True) # Ordena do melhor para o pior
        
        if random.random() < taxa_nao_melhor and len(competidores) > 1:
            # Com uma pequena chance, seleciona um dos outros competidores (não o melhor)
            return random.choice(competidores[1:]) 
        else:
            return competidores[0] # Seleciona o melhor
    
    def cruzamento_ponto_unico(self, pai1, pai2):
        """Realiza o cruzamento de ponto único entre dois pais."""
        n_genes = len(pai1.voos_ida)
        if n_genes == 0:
            return Cromossomo([], []), Cromossomo([], []) 
        
        ponto_corte = random.randint(1, n_genes - 1) # Ponto de corte entre 1 e n_genes-1

        # Cria os filhos trocando os segmentos de voos de ida e volta
        filho1_ida = pai1.voos_ida[:ponto_corte] + pai2.voos_ida[ponto_corte:]
        filho1_volta = pai1.voos_volta[:ponto_corte] + pai2.voos_volta[ponto_corte:]
        
        filho2_ida = pai2.voos_ida[:ponto_corte] + pai1.voos_ida[ponto_corte:]
        filho2_volta = pai2.voos_volta[:ponto_corte] + pai1.voos_volta[ponto_corte:]
        
        return (Cromossomo(filho1_ida, filho1_volta), 
                Cromossomo(filho2_ida, filho2_volta))
    
    def cruzamento_dois_pontos(self, pai1, pai2):
        """Realiza o cruzamento de dois pontos entre dois pais."""
        n_genes = len(pai1.voos_ida)
        if n_genes < 3: # Se não há genes suficientes para 2 pontos de corte, usa 1 ponto
            return self.cruzamento_ponto_unico(pai1, pai2)
            
        ponto1, ponto2 = sorted(random.sample(range(1, n_genes), 2)) # Dois pontos de corte distintos
        
        filho1_ida = pai1.voos_ida[:ponto1] + pai2.voos_ida[ponto1:ponto2] + pai1.voos_ida[ponto2:]
        filho1_volta = pai1.voos_volta[:ponto1] + pai2.voos_volta[ponto1:ponto2] + pai1.voos_volta[ponto2:]
        
        filho2_ida = pai2.voos_ida[:ponto1] + pai1.voos_ida[ponto1:ponto2] + pai2.voos_ida[ponto2:]
        filho2_volta = pai2.voos_volta[:ponto1] + pai1.voos_volta[ponto1:ponto2] + pai2.voos_volta[ponto2:]
        
        return (Cromossomo(filho1_ida, filho1_volta),
                Cromossomo(filho2_ida, filho2_volta))
    
    def cruzamento_uniforme(self, pai1, pai2):
        """Realiza o cruzamento uniforme entre dois pais (gene a gene)."""
        n_genes = len(pai1.voos_ida)
        if n_genes == 0:
            return Cromossomo([], []), Cromossomo([], []) 
        
        filho1_ida, filho1_volta = [], []
        filho2_ida, filho2_volta = [], []
        
        for i in range(n_genes):
            if random.random() < 0.5: # 50% de chance de pegar do pai1 ou pai2
                filho1_ida.append(pai1.voos_ida[i])
                filho1_volta.append(pai1.voos_volta[i])
                filho2_ida.append(pai2.voos_ida[i])
                filho2_volta.append(pai2.voos_volta[i])
            else:
                filho1_ida.append(pai2.voos_ida[i])
                filho1_volta.append(pai2.voos_volta[i])
                filho2_ida.append(pai1.voos_ida[i])
                filho2_volta.append(pai1.voos_volta[i])
        
        return (Cromossomo(filho1_ida, filho1_volta),
                Cromossomo(filho2_ida, filho2_volta))
    
    def cruzamento_baseado_custo(self, pai1, pai2):
        """
        Cruzamento que favorece voos mais baratos.
        Para cada gene (pessoa), o filho 1 recebe o voo mais barato entre os pais,
        e o filho 2 recebe o mais caro.
        """
        n_genes = len(pai1.voos_ida)
        if n_genes == 0:
            return Cromossomo([], []), Cromossomo([], []) 
        
        filho1_ida, filho1_volta = [], []
        filho2_ida, filho2_volta = [], []
        
        for i in range(n_genes):
            voo_ida_pai1, voo_ida_pai2 = pai1.voos_ida[i], pai2.voos_ida[i]
            voo_volta_pai1, voo_volta_pai2 = pai1.voos_volta[i], pai2.voos_volta[i]
            
            if voo_ida_pai1.preco <= voo_ida_pai2.preco:
                filho1_ida.append(voo_ida_pai1)
                filho2_ida.append(voo_ida_pai2)
            else:
                filho1_ida.append(voo_ida_pai2)
                filho2_ida.append(voo_ida_pai1)
            
            if voo_volta_pai1.preco <= voo_volta_pai2.preco:
                filho1_volta.append(voo_volta_pai1)
                filho2_volta.append(voo_volta_pai2)
            else:
                filho1_volta.append(voo_volta_pai2)
                filho2_volta.append(voo_volta_pai1)
        
        return (Cromossomo(filho1_ida, filho1_volta),
                Cromossomo(filho2_ida, filho2_volta))
    
    def cruzamento_baseado_horario(self, pai1, pai2):
        """
        Cruzamento que tenta sincronizar os horários.
        O filho para os voos de ida herda do pai com a menor "janela" de chegada.
        Para os voos de volta, faz um cruzamento uniforme.
        """
        n_genes = len(pai1.voos_ida)
        if n_genes == 0:
            return Cromossomo([], []), Cromossomo([], []) 
        
        horarios_chegada_pai1 = [voo.horario_chegada for voo in pai1.voos_ida]
        horarios_chegada_pai2 = [voo.horario_chegada for voo in pai2.voos_ida]
        
        if not horarios_chegada_pai1 or not horarios_chegada_pai2:
            return self.cruzamento_uniforme(pai1, pai2) # Fallback se não houver horários
        
        # Calcula o "span" de chegada para cada pai (diferença entre o último e o primeiro a chegar)
        span_pai1 = (max(horarios_chegada_pai1) - min(horarios_chegada_pai1)).total_seconds() / 3600
        span_pai2 = (max(horarios_chegada_pai2) - min(horarios_chegada_pai2)).total_seconds() / 3600
        
        # O filho de IDA herda os voos do pai que tem a menor dispersão de horários de chegada
        if span_pai1 <= span_pai2:
            filho1_ida = pai1.voos_ida.copy()
            filho2_ida = pai2.voos_ida.copy()
        else:
            filho1_ida = pai2.voos_ida.copy()
            filho2_ida = pai1.voos_ida.copy()
        
        # Para os voos de VOLTA, faz um cruzamento uniforme para manter a diversidade
        filho1_volta, filho2_volta = [], []
        for i in range(n_genes):
            if random.random() < 0.5:
                filho1_volta.append(pai1.voos_volta[i])
                filho2_volta.append(pai2.voos_volta[i])
            else:
                filho1_volta.append(pai2.voos_volta[i])
                filho2_volta.append(pai1.voos_volta[i])
        
        return (Cromossomo(filho1_ida, filho1_volta),
                Cromossomo(filho2_ida, filho2_volta))
    
    def mutacao_tradicional(self, cromossomo):
        """
        Aplica mutação tradicional: troca um voo por outro disponível aleatoriamente.
        """
        novo_cromossomo = copy.deepcopy(cromossomo)
        
        for i in range(len(novo_cromossomo.voos_ida)):
            pessoa_associada = self.pessoas[i] # Acessa a Pessoa para obter as opções de voos
            
            if random.random() < self.taxa_mutacao:
                if pessoa_associada.voos_ida: # Garante que há voos disponíveis para mutar
                    novo_cromossomo.voos_ida[i] = random.choice(pessoa_associada.voos_ida)
                
            if random.random() < self.taxa_mutacao:
                if pessoa_associada.voos_volta:
                    novo_cromossomo.voos_volta[i] = random.choice(pessoa_associada.voos_volta)
        
        return novo_cromossomo
    
    def mutacao_inteligente(self, cromossomo):
        """
        Aplica mutação de forma "inteligente", tentando reduzir o gap de horários.
        """
        novo_cromossomo = copy.deepcopy(cromossomo)
        
        if not self.pessoas: # Se não há pessoas, não há o que mutar
            return novo_cromossomo

        # Estratégia 1: Tenta reduzir o gap de chegada na ida
        horarios_chegada_ida = [voo.horario_chegada for voo in cromossomo.voos_ida]
        if len(horarios_chegada_ida) > 1:
            primeiro_chegada = min(horarios_chegada_ida)
            ultimo_chegada = max(horarios_chegada_ida)
            
            # Se o gap é significativo
            if (ultimo_chegada - primeiro_chegada).total_seconds() / 3600 > 1.0: # Ex: mais de 1 hora
                # Identifica a pessoa que chega mais tarde
                idx_mais_tarde = horarios_chegada_ida.index(ultimo_chegada)
                pessoa_alvo = self.pessoas[idx_mais_tarde]
                
                # Procura voos de ida mais cedo para essa pessoa que chega tarde
                voos_alternativos_ida = [
                    v for v in pessoa_alvo.voos_ida 
                    if v.horario_chegada < ultimo_chegada
                ]
                if voos_alternativos_ida and random.random() < 0.6: # 60% chance de aplicar
                    novo_cromossomo.voos_ida[idx_mais_tarde] = random.choice(voos_alternativos_ida)
        
        # Estratégia 2: Tenta reduzir o gap de saída na volta
        horarios_saida_volta = [voo.horario_saida for voo in cromossomo.voos_volta]
        if len(horarios_saida_volta) > 1:
            primeiro_saida = min(horarios_saida_volta)
            ultimo_saida = max(horarios_saida_volta)

            if (ultimo_saida - primeiro_saida).total_seconds() / 3600 > 1.0:
                # Identifica a pessoa que sai mais cedo
                idx_mais_cedo = horarios_saida_volta.index(primeiro_saida)
                pessoa_alvo = self.pessoas[idx_mais_cedo]

                # Procura voos de volta mais tarde para essa pessoa que sai cedo
                voos_alternativos_volta = [
                    v for v in pessoa_alvo.voos_volta 
                    if v.horario_saida > primeiro_saida
                ]
                if voos_alternativos_volta and random.random() < 0.6:
                    novo_cromossomo.voos_volta[idx_mais_cedo] = random.choice(voos_alternativos_volta)
        
        # Estratégia 3: Mutação aleatória simples (com menor probabilidade)
        for i in range(len(novo_cromossomo.voos_ida)):
            pessoa_associada = self.pessoas[i]
            if random.random() < self.taxa_mutacao * 0.2: # Taxa reduzida para mutação aleatória
                if pessoa_associada.voos_ida:
                    novo_cromossomo.voos_ida[i] = random.choice(pessoa_associada.voos_ida)
            
            if random.random() < self.taxa_mutacao * 0.2:
                if pessoa_associada.voos_volta:
                    novo_cromossomo.voos_volta[i] = random.choice(pessoa_associada.voos_volta)
        
        return novo_cromossomo
    
    def evoluir_geracao(self, tipo_cruzamento='ponto_unico', usar_mutacao_inteligente=False):
        """
        Cria a próxima geração da população através de elitismo, seleção, cruzamento e mutação.
        """
        nova_populacao = []
        
        # Elitismo: Preserva os melhores cromossomos diretamente para a próxima geração
        self.populacao.sort(key=lambda x: x.fitness, reverse=True)
        elite_size = int(self.taxa_elite * self.tamanho_populacao)
        nova_populacao.extend(self.populacao[:elite_size])
        
        # Preenche o restante da nova população com filhos gerados por cruzamento e mutação
        while len(nova_populacao) < self.tamanho_populacao:
            # Seleção de pais
            pai1 = self.selecao_torneio()
            pai2 = self.selecao_torneio()
            
            # Garante que os pais são válidos e diferentes, se possível
            if pai1 is None or pai2 is None: # Se a população estiver vazia ou com poucos indivíduos
                break # Sai do loop se não conseguir pais válidos
            if pai1 == pai2 and len(self.populacao) > 1: # Tenta pegar pais diferentes
                pai2 = self.selecao_torneio()
            
            # Escolhe o tipo de cruzamento
            if tipo_cruzamento == 'ponto_unico':
                filho1, filho2 = self.cruzamento_ponto_unico(pai1, pai2)
            elif tipo_cruzamento == 'dois_pontos':
                filho1, filho2 = self.cruzamento_dois_pontos(pai1, pai2)
            elif tipo_cruzamento == 'uniforme':
                filho1, filho2 = self.cruzamento_uniforme(pai1, pai2)
            elif tipo_cruzamento == 'custo':
                filho1, filho2 = self.cruzamento_baseado_custo(pai1, pai2)
            elif tipo_cruzamento == 'horario':
                filho1, filho2 = self.cruzamento_baseado_horario(pai1, pai2)
            else: # Padrão
                filho1, filho2 = self.cruzamento_uniforme(pai1, pai2)
            
            # Aplica mutação aos filhos
            if usar_mutacao_inteligente:
                filho1 = self.mutacao_inteligente(filho1)
                filho2 = self.mutacao_inteligente(filho2)
            else:
                filho1 = self.mutacao_tradicional(filho1)
                filho2 = self.mutacao_tradicional(filho2)
            
            nova_populacao.extend([filho1, filho2])
        
        # Atualiza a população, garantindo o tamanho correto
        self.populacao = nova_populacao[:self.tamanho_populacao]
    
    def executar(self, max_geracoes=1000, tipo_cruzamento='ponto_unico', 
                 usar_mutacao_inteligente=False, verbose=True):
        """
        Executa o algoritmo genético pelo número especificado de gerações.
        Retorna a melhor solução encontrada e os históricos de fitness/custo.
        """
        if not self.pessoas:
            print("Erro: Nenhuma pessoa configurada para o algoritmo genético. Verifique o carregamento dos dados.")
            return None, {'fitness': [], 'custo': []}

        # Gera a população inicial aleatoriamente
        try:
            self.gerar_populacao_inicial()
        except ValueError as e:
            print(f"Erro ao iniciar população: {e}. Verifique se todas as pessoas têm voos disponíveis.")
            return None, {'fitness': [], 'custo': []}

        historico_fitness = []  
        historico_custo = []    
        
        # print(f"\nIniciando execução do AG com {len(self.pessoas)} pessoas...") # Comentado para evitar log repetitivo em testes

        for geracao in range(max_geracoes):
            self.avaliar_populacao() # Avalia todos os cromossomos na geração
            
            # Encontra o melhor cromossomo da geração atual
            melhor_atual = max(self.populacao, key=lambda x: x.fitness)
            historico_fitness.append(melhor_atual.fitness)
            
            # Calcula o custo total do melhor cromossomo atual para exibição
            custo_melhor_atual = sum(voo.preco for voo in melhor_atual.voos_ida + melhor_atual.voos_volta)
            historico_custo.append(custo_melhor_atual) # Adiciona o custo do melhor cromossomo, não o custo médio da população

            if melhor_atual.fitness > self.melhor_fitness:
                self.melhor_fitness = melhor_atual.fitness
                self.geracoes_sem_melhoria = 0 # Reseta o contador se houver melhoria
                if verbose:
                    print(f"Geração {geracao:3d}: Fitness = {self.melhor_fitness:8.2f} | Custo = €{custo_melhor_atual:7.0f}")
            else:
                self.geracoes_sem_melhoria += 1
                # if verbose and geracao % 10 == 0: # Imprime menos frequentemente se não houver melhoria - Comentado para testes
                #      print(f"Geração {geracao:3d}: Fitness = {melhor_atual.fitness:8.2f} (sem melhoria, {self.geracoes_sem_melhoria} de {self.geracoes_sem_melhoria_max})")
            
            # Critério de parada por convergência (melhoria percentual muito baixa)
            if self.criterio_parada_avancado(historico_fitness):
                if verbose:
                    print(f"--- Parada por convergência estatística na geração {geracao} ---")
                break
            
            # Critério de parada por limite de gerações sem melhoria
            if self.geracoes_sem_melhoria >= self.geracoes_sem_melhoria_max:
                if verbose:
                    print(f"--- Parada por limite de gerações sem melhoria na geração {geracao} ---")
                break
            
            # Evolui para a próxima geração
            self.evoluir_geracao(tipo_cruzamento, usar_mutacao_inteligente)
        
        # Garante que a melhor solução final é avaliada e retornada
        self.avaliar_populacao() 
        melhor_solucao = max(self.populacao, key=lambda x: x.fitness)
        return melhor_solucao, {'fitness': historico_fitness, 'custo': historico_custo}
    
    def criterio_parada_avancado(self, historico_fitness, janela=20, tolerancia=0.001):
        """
        Critério de parada: o algoritmo para se a melhoria no fitness
        nas últimas 'janela' gerações for menor que 'tolerancia' (0.1%).
        """
        if len(historico_fitness) < janela:
            return False
        
        fitness_atual = historico_fitness[-1]
        fitness_anterior = historico_fitness[-janela]
        
        # Evita divisão por zero ou por valores muito pequenos
        if abs(fitness_anterior) < 1e-9: 
            return abs(fitness_atual - fitness_anterior) < (tolerancia * abs(fitness_anterior) + 1e-9)
        
        melhoria_percentual = abs((fitness_atual - fitness_anterior) / fitness_anterior)
        return melhoria_percentual < tolerancia
    
    def mostrar_detalhes_solucao(self, cromossomo):
        """Exibe os detalhes da melhor solução encontrada de forma legível."""
        print("\n" + "="*70)
        print("DETALHES DA MELHOR SOLUÇÃO ENCONTRADA")
        print("="*70)
        
        custo_total = sum(voo.preco for voo in cromossomo.voos_ida + cromossomo.voos_volta)
        print(f"Custo Total dos Voos: €{custo_total:.2f}")
        
        # Análise dos voos de ida
        print(f"\n--- VOOS DE IDA PARA {self.destino} ---")
        horarios_chegada_ida = []
        for i, voo in enumerate(cromossomo.voos_ida):
            pessoa = self.pessoas[i]
            print(f"  {pessoa.nome:<15} | Voo: {voo}")
            horarios_chegada_ida.append(voo.horario_chegada)
        
        if horarios_chegada_ida:
            primeiro_chegada = min(horarios_chegada_ida)
            ultimo_chegada = max(horarios_chegada_ida)
            gap_ida = (ultimo_chegada - primeiro_chegada).total_seconds() / 3600
            
            print(f"\n  Chegada do 1º: {primeiro_chegada.strftime('%H:%M')}")
            print(f"  Chegada do Último: {ultimo_chegada.strftime('%H:%M')}")
            print(f"  Gap de Chegada (espera da van): {gap_ida:.1f} horas")
        else:
            print("  Nenhum voo de ida selecionado ou pessoas ausentes.")
            
        # Análise dos voos de volta
        print(f"\n--- VOOS DE VOLTA DE {self.destino} ---")
        horarios_saida_volta = []
        for i, voo in enumerate(cromossomo.voos_volta):
            pessoa = self.pessoas[i]
            print(f"  {pessoa.nome:<15} | Voo: {voo}")
            horarios_saida_volta.append(voo.horario_saida)
        
        if horarios_saida_volta:
            primeiro_saida = min(horarios_saida_volta)
            ultimo_saida = max(horarios_saida_volta)
            gap_volta = (ultimo_saida - primeiro_saida).total_seconds() / 3600
            
            print(f"\n  Saída do 1º: {primeiro_saida.strftime('%H:%M')}")
            print(f"  Saída do Último: {ultimo_saida.strftime('%H:%M')}")
            print(f"  Gap de Saída (janela da van): {gap_volta:.1f} horas")
        else:
            print("  Nenhum voo de volta selecionado ou pessoas ausentes.")
            
        print(f"\nFitness Total da Solução: {cromossomo.fitness:.2f}")
        print("="*70)

def plotar_historicos(resultados, titulo_sufixo=""):
    """
    Plota os históricos de fitness e custo para cada teste em figuras separadas.
    Cada teste (configuração de cruzamento/mutação) terá sua própria figura com dois subplots.
    """
    if not resultados:
        print("Nenhum resultado para plotar.")
        return

    for nome_teste, resultado in resultados.items():
        historicos = resultado['historicos']
        solucao = resultado['solucao']

        # Verifica se há dados válidos para plotar este teste específico
        if not historicos or not historicos['fitness'] or not historicos['custo'] or solucao is None:
            print(f"Pulando plot para '{nome_teste}' devido à falta de dados.")
            continue

        # Cria uma NOVA figura para cada teste
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6)) # Uma linha, duas colunas para fitness e custo
        
        # Título geral para a figura atual
        fig.suptitle(f'Históricos de Fitness e Custo para: {nome_teste} {titulo_sufixo}', fontsize=16)

        # Plotar Fitness
        ax_fitness = axes[0]
        ax_fitness.plot(historicos['fitness'], label='Melhor Fitness', color='blue')
        ax_fitness.set_title('Evolução do Fitness', fontsize=12)
        ax_fitness.set_xlabel('Geração', fontsize=10)
        ax_fitness.set_ylabel('Fitness', fontsize=10)
        ax_fitness.grid(True, linestyle='--', alpha=0.7)
        ax_fitness.legend(fontsize=9)

        # Plotar Custo
        ax_cost = axes[1]
        ax_cost.plot(historicos['custo'], label='Custo da Melhor Solução', color='red')
        ax_cost.set_title('Evolução do Custo', fontsize=12)
        ax_cost.set_xlabel('Geração', fontsize=10)
        ax_cost.set_ylabel('Custo (€)', fontsize=10)
        ax_cost.grid(True, linestyle='--', alpha=0.7)
        ax_cost.legend(fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para evitar sobreposição do título
        plt.show() # Mostra a figura atual

def executar_e_coletar(pessoas, config, num_runs=30):
    """Executa o AG múltiplas vezes para uma dada configuração e retorna a média do melhor fitness."""
    resultados_runs = []
    
    for _ in range(num_runs):
        ag_run = AlgoritmoGenetico(
            pessoas=copy.deepcopy(pessoas), # Sempre faz deepcopy para isolamento das runs
            destino=config.get('destino', 'FCO'),
            tamanho_populacao=config.get('tamanho_populacao', 80),
            taxa_mutacao=config.get('taxa_mutacao', 0.12),
            taxa_elite=config.get('taxa_elite', 0.08),
            geracoes_sem_melhoria_max=config.get('geracoes_sem_melhoria_max', 40),
            peso_custo=config.get('peso_custo', 1.0),
            peso_gap_chegada=config.get('peso_gap_chegada', 100.0),
            peso_gap_saida=config.get('peso_gap_saida', 100.0)
        )
        melhor_solucao, _ = ag_run.executar(
            max_geracoes=config.get('max_geracoes', 150),
            tipo_cruzamento=config.get('tipo_cruzamento', 'uniforme'), # Pode ser variado também aqui se for o caso
            usar_mutacao_inteligente=config.get('usar_mutacao_inteligente', False), # Pode ser variado também
            verbose=False # Desativa verbose para rodadas de teste
        )
        if melhor_solucao:
            resultados_runs.append(melhor_solucao.fitness)
    
    if resultados_runs:
        return sum(resultados_runs) / len(resultados_runs)
    return float('-inf') # Se nenhuma solução válida for encontrada

def main():
    print("=== Algoritmo Genético para Otimização de Voos para Encontro ===\n")
    print("Objetivo: Encontrar os melhores voos de ida e volta para um grupo de pessoas,")
    print("minimizando custo total e tempo de espera da van em Roma (FCO).\n")
    
    # 1. Carregar as pessoas e seus voos disponíveis
    # Usamos uma instância temporária do AG apenas para carregar os dados
    temp_loader_ag = AlgoritmoGenetico(pessoas=[], destino='FCO') 
    
    pessoas_otimizar = []
    try:
        pessoas_otimizar = temp_loader_ag.criar_pessoas_from_arquivo('flights.txt')
        if not pessoas_otimizar:
            print("\nAVISO: Nenhuma pessoa válida encontrada no 'flights.txt' com voos de ida e volta.")
            print("Gerando dados de exemplo para continuar.")
            # Se o arquivo não fornecer dados válidos, usamos os dados de exemplo
            example_voos_data = temp_loader_ag.gerar_dados_exemplo()
            
            # Recria a lista de pessoas a partir dos dados de exemplo
            for origem_cidade, voos_dict in example_voos_data.items():
                if origem_cidade != temp_loader_ag.destino and voos_dict['ida'] and voos_dict['volta']:
                    nome_pessoa = f"Pessoa_{origem_cidade}"
                    pessoas_otimizar.append(Pessoa(nome_pessoa, origem_cidade, voos_dict['ida'], voos_dict['volta']))
            
            if not pessoas_otimizar:
                print("ERRO CRÍTICO: Não foi possível gerar pessoas válidas com dados de exemplo. Encerrando.")
                return
            
    except Exception as e:
        print(f"\nERRO: Falha ao carregar 'flights.txt' ({e}). Gerando dados de exemplo para continuar.")
        # Se houver qualquer erro no carregamento do arquivo, usa os dados de exemplo
        example_voos_data = temp_loader_ag.gerar_dados_exemplo()
        
        for origem_cidade, voos_dict in example_voos_data.items():
            if origem_cidade != temp_loader_ag.destino and voos_dict['ida'] and voos_dict['volta']:
                nome_pessoa = f"Pessoa_{origem_cidade}"
                pessoas_otimizar.append(Pessoa(nome_pessoa, origem_cidade, voos_dict['ida'], voos_dict['volta']))
        
        if not pessoas_otimizar:
            print("ERRO CRÍTICO: Não foi possível gerar pessoas válidas com dados de exemplo. Encerrando.")
            return

    # 2. Configurar o Algoritmo Genético com as pessoas carregadas (configurações padrão para testes)
    ag_base_config = {
        'pessoas': pessoas_otimizar, 
        'destino': 'FCO', 
        'tamanho_populacao': 80,
        'taxa_mutacao': 0.12,
        'taxa_elite': 0.08,
        'geracoes_sem_melhoria_max': 40,
        'max_geracoes': 150,
        'tipo_cruzamento': 'uniforme', # Define um tipo de cruzamento base
        'usar_mutacao_inteligente': False, # Define uma mutação base
        'verbose': True # Ativado para o primeiro conjunto de testes
    }

    # Instância principal do AG para mostrar detalhes finais e usar como base para outros testes
    ag_main = AlgoritmoGenetico(
        pessoas=copy.deepcopy(ag_base_config['pessoas']), 
        destino=ag_base_config['destino'], 
        tamanho_populacao=ag_base_config['tamanho_populacao'],
        taxa_mutacao=ag_base_config['taxa_mutacao'],
        taxa_elite=ag_base_config['taxa_elite'],
        geracoes_sem_melhoria_max=ag_base_config['geracoes_sem_melhoria_max']
    )

    print(f"\n--- Configuração da Execução Principal ---")
    print(f"Número de Pessoas a Otimizar: {len(ag_main.pessoas)}")
    print(f"Aeroporto de Encontro: {ag_main.destino}")
    if ag_main.pessoas:
        print(f"Exemplo: {ag_main.pessoas[0].nome} de {ag_main.pessoas[0].origem_cidade} tem {len(ag_main.pessoas[0].voos_ida)} voos de ida e {len(ag_main.pessoas[0].voos_volta)} voos de volta disponíveis.")
    print(f"Tamanho da População: {ag_main.tamanho_populacao}")
    print(f"Taxa de Mutação: {ag_main.taxa_mutacao*100:.0f}%")
    print(f"Taxa de Elitismo: {ag_main.taxa_elite*100:.0f}%")
    print(f"Gerações sem Melhoria para Parada: {ag_main.geracoes_sem_melhoria_max}")

    # 3. Testar diferentes configurações de cruzamento e mutação
    tipos_cruzamento = ['ponto_unico', 'dois_pontos', 'uniforme', 'custo', 'horario']
    tipos_mutacao = [False, True] # False para Tradicional, True para Inteligente
    
    melhores_resultados_testes = {}
    
    for tipo_cruz in tipos_cruzamento:
        for usar_mutacao_inteligente in tipos_mutacao:
            nome_teste = f"Cruzamento:{tipo_cruz.replace('_', ' ').title()} + Mutação:{'Inteligente' if usar_mutacao_inteligente else 'Tradicional'}"
            
            print(f"\n\n{'='*80}")
            print(f"INICIANDO TESTE: {nome_teste.upper()}")
            print(f"{'='*80}")
            
            # Cria uma NOVA instância do AG para cada teste para garantir resultados independentes
            # Faz um deepcopy das pessoas para que as instâncias do AG não alterem os dados originais
            ag_para_teste = AlgoritmoGenetico(
                pessoas=copy.deepcopy(ag_base_config['pessoas']), 
                destino=ag_base_config['destino'], 
                tamanho_populacao=ag_base_config['tamanho_populacao'],
                taxa_mutacao=ag_base_config['taxa_mutacao'],
                taxa_elite=ag_base_config['taxa_elite'],
                geracoes_sem_melhoria_max=ag_base_config['geracoes_sem_melhoria_max']
            )

            melhor_solucao_teste, historicos_teste = ag_para_teste.executar(
                max_geracoes=ag_base_config['max_geracoes'], 
                tipo_cruzamento=tipo_cruz,
                usar_mutacao_inteligente=usar_mutacao_inteligente,
                verbose=True
            )
            
            if melhor_solucao_teste:
                melhores_resultados_testes[nome_teste] = {
                    'solucao': melhor_solucao_teste,
                    'historicos': historicos_teste
                }
                ag_para_teste.mostrar_detalhes_solucao(melhor_solucao_teste)
            else:
                print(f"Nenhuma solução válida encontrada para o teste: {nome_teste}.")
    
    # 4. Análise Comparativa Final dos Testes Iniciais
    print(f"\n\n{'#'*90}")
    print(f"{'':^90}") # Centraliza o texto
    print(f"{'ANÁLISE COMPARATIVA DOS TESTES DE ALGORITMO GENÉTICO':^90}")
    print(f"{'':^90}")
    print(f"{'#'*90}")
    
    # Cabeçalho da tabela
    print(f"\n{'Configuração':<45} | {'Fitness':>10} | {'Custo Total (€)':>15} | {'Gap Ida (h)':>13} | {'Gap Volta (h)':>15}")
    print("-" * 100)
    
    if not melhores_resultados_testes:
        print("Nenhum resultado para exibir na análise comparativa.")
    else:
        # Exibe os resultados de cada teste
        for nome, resultado in melhores_resultados_testes.items():
            solucao = resultado['solucao']
            if solucao: # Garante que há uma solução válida
                custo = sum(voo.preco for voo in solucao.voos_ida + solucao.voos_volta)
                
                horarios_chegada = [voo.horario_chegada for voo in solucao.voos_ida]
                gap_ida = (max(horarios_chegada) - min(horarios_chegada)).total_seconds() / 3600 if horarios_chegada else 0
                
                horarios_saida = [voo.horario_saida for voo in solucao.voos_volta]
                gap_volta = (max(horarios_saida) - min(horarios_saida)).total_seconds() / 3600 if horarios_saida else 0
                
                print(f"{nome:<45} | {solucao.fitness:10.2f} | {custo:15.2f} | {gap_ida:13.1f} | {gap_volta:15.1f}")
            else:
                print(f"{nome:<45} | {'N/A':>10} | {'N/A':>15} | {'N/A':>13} | {'N/A':>15}")
        
        # Encontra e destaca a melhor configuração geral
        valid_results = {k: v for k, v in melhores_resultados_testes.items() if v['solucao'] is not None}
        
        if valid_results:
            melhor_config_geral = max(valid_results.items(), key=lambda x: x[1]['solucao'].fitness)
            print(f"\n\n{'*'*90}")
            print(f"🏆 MELHOR CONFIGURAÇÃO GERAL ENCONTRADA: {melhor_config_geral[0].upper()}")
            print(f"   Fitness da Melhor Solução: {melhor_config_geral[1]['solucao'].fitness:.2f}")
            print(f"{'*'*90}")
            # Chama mostrar_detalhes_solucao usando a instância ag_main
            ag_main.mostrar_detalhes_solucao(melhor_config_geral[1]['solucao'])
        else:
            print("\nInfelizmente, nenhuma solução válida foi encontrada em nenhum dos testes.")
    
    # 5. Plotar os históricos de fitness e custo para cada teste (Item 1 do pedido do usuário)
    if melhores_resultados_testes:
        print("\n\n--- Gerando Gráficos de Histórico de Fitness e Custo para cada Configuração ---")
        plotar_historicos(melhores_resultados_testes)

    # 6. Testar variações nos pesos da função de fitness (Item 2 do pedido do usuário)
    print(f"\n\n{'#'*90}")
    print(f"{'':^90}")
    print(f"{'ANÁLISE DE VARIAÇÃO DE PESOS NA FUNÇÃO DE FITNESS':^90}")
    print(f"{'':^90}")
    print(f"{'#'*90}")

    pesos_custo_variados = [0.5, 1.0, 2.0] # Menos custo, Custo padrão, Mais custo
    pesos_gap_variados = [50.0, 100.0, 200.0] # Menos penalty, Penalty padrão, Mais penalty

    resultados_pesos = {}
    default_config = copy.deepcopy(ag_base_config)
    default_config['verbose'] = False # Desativa o verbose para esta fase

    for pc in pesos_custo_variados:
        for pg in pesos_gap_variados:
            # Usar o mesmo peso para gap de chegada e saída para simplificar a análise inicial
            nome_config = f"Custo:{pc:.1f}, Gap:{pg:.1f}"
            print(f"\nTestando configuração de pesos: {nome_config}")
            
            config_atual = default_config.copy()
            config_atual['peso_custo'] = pc
            config_atual['peso_gap_chegada'] = pg
            config_atual['peso_gap_saida'] = pg # Mantém igual por agora
            
            # Executa múltiplas vezes e coleta a média do fitness
            avg_fitness = executar_e_coletar(pessoas_otimizar, config_atual, num_runs=5) 
            resultados_pesos[nome_config] = avg_fitness

    # Plotar resultados da variação de pesos
    if resultados_pesos:
        labels = []
        fitness_values = []
        for label, fitness in resultados_pesos.items():
            labels.append(label)
            fitness_values.append(fitness)

        plt.figure(figsize=(12, 7))
        plt.bar(labels, fitness_values, color='skyblue')
        plt.xlabel('Configuração de Pesos (Peso Custo, Peso Gap)', fontsize=12)
        plt.ylabel('Média do Melhor Fitness', fontsize=12)
        plt.title('Impacto da Variação dos Pesos na Função de Fitness', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # 7. Testar variações nos parâmetros do AG (Item 3 do pedido do usuário)
    print(f"\n\n{'#'*90}")
    print(f"{'':^90}")
    print(f"{'ANÁLISE DE VARIAÇÃO DE PARÂMETROS DO ALGORITMO GENÉTICO':^90}")
    print(f"{'':^90}")
    print(f"{'#'*90}")

    # Variando um parâmetro por vez (mantendo os outros como base_config)
    
    # Variação do Tamanho da População
    tamanhos_populacao = [50, 80, 150] # Menor, Padrão, Maior
    resultados_tamanho_pop = {}
    for tp in tamanhos_populacao:
        config_atual = default_config.copy()
        config_atual['tamanho_populacao'] = tp
        print(f"\nTestando Tamanho da População: {tp}")
        avg_fitness = executar_e_coletar(pessoas_otimizar, config_atual, num_runs=5)
        resultados_tamanho_pop[f"População:{tp}"] = avg_fitness

    plt.figure(figsize=(8, 5))
    plt.bar(list(resultados_tamanho_pop.keys()), list(resultados_tamanho_pop.values()), color='lightcoral')
    plt.xlabel('Tamanho da População', fontsize=12)
    plt.ylabel('Média do Melhor Fitness', fontsize=12)
    plt.title('Impacto do Tamanho da População no Fitness', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Variação da Taxa de Mutação
    taxas_mutacao = [0.05, 0.12, 0.20] # Menor, Padrão, Maior
    resultados_taxa_mut = {}
    for tm in taxas_mutacao:
        config_atual = default_config.copy()
        config_atual['taxa_mutacao'] = tm
        print(f"\nTestando Taxa de Mutação: {tm}")
        avg_fitness = executar_e_coletar(pessoas_otimizar, config_atual, num_runs=5)
        resultados_taxa_mut[f"Mutação:{tm*100:.0f}%"] = avg_fitness

    plt.figure(figsize=(8, 5))
    plt.bar(list(resultados_taxa_mut.keys()), list(resultados_taxa_mut.values()), color='lightgreen')
    plt.xlabel('Taxa de Mutação', fontsize=12)
    plt.ylabel('Média do Melhor Fitness', fontsize=12)
    plt.title('Impacto da Taxa de Mutação no Fitness', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Variação da Taxa de Elite
    taxas_elite = [0.02, 0.08, 0.15] # Menor, Padrão, Maior
    resultados_taxa_elite = {}
    for te in taxas_elite:
        config_atual = default_config.copy()
        config_atual['taxa_elite'] = te
        print(f"\nTestando Taxa de Elite: {te}")
        avg_fitness = executar_e_coletar(pessoas_otimizar, config_atual, num_runs=5)
        resultados_taxa_elite[f"Elite:{te*100:.0f}%"] = avg_fitness

    plt.figure(figsize=(8, 5))
    plt.bar(list(resultados_taxa_elite.keys()), list(resultados_taxa_elite.values()), color='lightsalmon')
    plt.xlabel('Taxa de Elite', fontsize=12)
    plt.ylabel('Média do Melhor Fitness', fontsize=12)
    plt.title('Impacto da Taxa de Elite no Fitness', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()