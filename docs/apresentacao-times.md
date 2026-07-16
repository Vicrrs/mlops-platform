# Roteiro de apresentação — por que um repositório central de MLOps (e por que cada time também precisa do seu)

Roteiro para apresentar a proposta aos times de dados/ML e a lideranças. ~12-15 min de fala + perguntas. Linguagem propositalmente simples — sem jargão de plataforma, sem citar nomes de arquivos.

---

## 1. Abertura — o gancho (1 min)

> "Quero fazer uma pergunta antes de começar: quantos de vocês, ao subir um modelo pra produção, já pararam pra pensar — 'como eu sei que esse modelo é melhor que o anterior?', 'quem aprovou isso?', 'se der problema às 3h da manhã, como eu volto pro modelo antigo rápido?'"

Deixa a pergunta no ar por 2 segundos. A ideia é que ninguém tenha uma resposta rápida e uniforme — e é exatamente esse o problema que vamos resolver.

## 2. O problema, em uma frase (1 min)

> "Hoje, cada time de dados resolve os mesmos problemas — testar, validar, comparar modelo novo com o antigo, aprovar produção, reverter se der ruim — do zero, cada um do seu jeito. Isso significa que a gente reinventa a roda dez vezes, e nenhuma das dez rodas tem a mesma qualidade."

Não é sobre time nenhum estar fazendo errado. É que ninguém devia estar resolvendo esse problema sozinho.

## 3. A proposta, em uma frase (1 min)

> "A proposta é separar duas coisas que hoje estão misturadas: o que é **igual pra qualquer modelo** (testar, validar, comparar, aprovar, reverter) e o que é **específico de cada modelo** (os dados, as features, o algoritmo)."

- O que é igual → vira um **repositório central de MLOps**.
- O que é específico → continua no **repositório de cada time**, um por projeto/modelo.

Nenhum código de modelo se muda de lugar. Só a "regra do jogo" fica compartilhada.

## 4. Por que um repositório central específico (a defesa) (4 min)

Três argumentos, nessa ordem — do mais concreto pro mais estratégico:

**a) Corrigir uma vez, proteger todo mundo.**
> "Se a gente encontra uma falha de segurança ou um bug no jeito de validar dados, hoje eu preciso avisar dez times e torcer pra cada um corrigir do jeito certo. Com um repositório central, eu corrijo uma vez, e todo projeto que atualizar a versão já está protegido."

**b) Todo modelo passa pelo mesmo processo de confiança.**
> "Quando alguém de compliance, de risco, ou um cliente perguntar 'como vocês garantem que um modelo em produção foi testado e aprovado direito', a resposta precisa ser uma só — não dez respostas diferentes dependendo de quem você pergunta."

**c) Nenhum time perde tempo reinventando infraestrutura.**
> "Hoje, todo projeto novo gasta as primeiras semanas resolvendo os mesmos problemas de novo: como estruturar o pipeline, como escrever os testes, como comparar modelo novo com o antigo. Com um ponto de partida pronto, essas semanas viram horas."

Se quiser uma analogia rápida: *"É a diferença entre cada time construir a própria estrada até o mercado, e todo mundo usar a mesma rodovia — construída e mantida por quem entende de rodovia."*

## 5. Por que cada time AINDA precisa do seu próprio repositório (3 min)

Esse é o ponto que geralmente gera reação de "então vai virar tudo centralizado e travado?" — antecipe isso direto:

> "Só pra ser bem claro: eu não estou propondo que todo mundo trabalhe num repositório gigante e único. Pelo contrário."

Três razões:

**a) O time é dono do próprio domínio.**
> "O time de crédito entende de risco de crédito. O time de churn entende de comportamento de cliente. Ninguém de fora deveria precisar aprovar uma mudança de feature ou de algoritmo — isso é trabalho de vocês, e vocês precisam poder iterar rápido, sem depender de mais ninguém."

**b) Um repositório único vira gargalo.**
> "Se todo mundo estiver no mesmo repositório, o pipeline de um time trava o do outro, os testes de um projeto rodam mesmo quando só o outro mudou, e ninguém consegue rodar nada localmente sem carregar o código de dez modelos diferentes que não tem nada a ver com o dele."

**c) Dado sensível não deveria estar visível pra quem não precisa dele.**
> "Dados de crédito e dados de churn não têm por que estar no mesmo lugar, visíveis pras mesmas pessoas. Repositório separado por projeto também é uma questão de permissão e de segurança."

## 6. Como fica na prática, sem jargão (2 min)

> "Na prática, funciona assim: o time de MLOps mantém as regras do jogo. Cada time de dados copia um modelo pronto pra começar um projeto novo — já vem com a estrutura, os testes obrigatórios, e o fluxo de aprovação configurados. O arquivo de pipeline de cada projeto é pequeno, tem uns 40 linhas, e só importa as regras do repositório central."

> "E o mais importante: cada projeto escolhe *quando* atualizar pra uma versão nova das regras — ninguém é pego de surpresa porque alguém mudou algo no central. A versão fica travada até o time decidir atualizar."

## 7. Prova — isso não é teoria (2 min)

> "Eu já implementei e testei isso de ponta a ponta, localmente, sem gastar um centavo de infraestrutura: treinei um modelo real, registrei, comparei com uma versão anterior, promovi pra 'produção', e depois revertei — tudo automatizado, com 107 testes passando. Não é uma proposta no papel, é algo que já roda."

Se fizer sentido no momento, mostre rapidamente o diagrama da arquitetura (`docs/architecture.drawio`) como apoio visual — mas o argumento já foi feito de boca.

## 8. Perguntas que vão surgir (tenha a resposta pronta)

| Pergunta | Resposta curta |
|---|---|
| "Isso não vai deixar a gente mais lento?" | "O contrário — hoje vocês perdem semanas reinventando isso a cada projeto novo. Um ponto de partida pronto poupa esse tempo." |
| "E se o time central quebrar algo e meu projeto parar de funcionar?" | "Não quebra — cada projeto trava numa versão específica das regras. Vocês decidem quando atualizar." |
| "Por que não cada um faz do seu jeito, como sempre foi?" | "Porque hoje, se um modelo falha em produção, não existe uma forma única de saber se ele foi testado direito, quem aprovou, e como reverter rápido. Isso é risco da empresa, não só do time." |
| "Quem vai manter esse repositório central?" | "O time de MLOps — é literalmente pra isso que o time existe." |
| "Vou perder autonomia sobre o meu modelo?" | "Zero. Vocês continuam donos de dados, features e algoritmo. A única coisa compartilhada é o processo de validar/aprovar/reverter." |

## 9. Fechamento — o pedido concreto (1 min)

> "O pedido é simples: aprovação pra criar o repositório central, e um sprint pra migrar um projeto piloto pra esse modelo. A gente mede o ganho nesse piloto, e só depois decide expandir pros outros times."

Termine sempre com um pedido claro e pequeno — não "adotem tudo agora", e sim "deixa eu provar com um projeto".

---

### Se for virar slide

1. O problema (a pergunta de abertura)
2. Dois repositórios, duas responsabilidades (diagrama da página 1 do `.drawio`)
3. Por que centralizar as regras (3 bullets da seção 4)
4. Por que não centralizar o código (3 bullets da seção 5)
5. Como fica na prática (diagrama da página 3 — pipeline)
6. Prova: números reais do piloto já testado
7. Pedido: aprovar o repo central + 1 sprint de piloto
