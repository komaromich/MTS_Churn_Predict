# MTS_HW_2
Второе домашнее задание команды Dungeon machines в рамках школы МТС.Тета. 

# Краткий отчёт

Поставлена бизнес-цель уменьшение оттока клиентов, для достижения цели была решена задачу бинарной классификации. \n
Был выбран датасет в сфере телеком. На основании этого датасета было проведёно предпроектное исследование. 
В датасете обнаружен дисбаланс классов, характерный для задачи оттока. Данные становятся достаточно хорошо разделимы нелинейной поверхностью. Были построены две baseline -  модели.
При увеличении ROC-AUC на 10% эконом. эффект вырос на 275%.


# Источник данных

https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383

# Описание проекта

Вам предоставляется отчет и набор данных от телекоммуникационной компании. Данные содержат информацию о более чем трех тысячах пользователей, их демографических характеристиках, услугах, которыми они пользуются, длительности использования услуг оператора и сумме оплаты.

Задача - проанализировать данные и спрогнозировать отток пользователей (выявить людей, которые будут и не будут продлевать свой контракт). Удержание пользователей - одна из наиболее актуальных задач в областях, где распространение услуги составляет порядка 100%. Ярким примером такой области является сфера телекома. Удержание клиента обойдется компании дешевле, чем привлечение нового.

Прогнозируя отток, мы можем вовремя среагировать и постараться удержать клиента, который хочет уйти. На основании данных об услугах, которыми пользуется клиент, мы можем сделать ему специальное предложение, пытаясь изменить его решение покинуть оператора. Это сделает задачу удержания проще в реализации, чем задачу привлечения новых пользователей. На основе полученных данных команда DS проверяет возможно и эффективно ли строить и применять модель оттока. В положительном случае модель обучается, проводится A/B тест. Если результаты A/B теста экономически и статистически значимы, то принимается решение о внедрении модели в продакшн. Команде разработки даётся задание подготовить соответствующий функционал в приложении, а также автоматическую рассылку на почту и телефон. После этого выбирается периодичность с которой будет производится анализ на предмет оттока. В результате этого на серверах компании появляется соответствующая информация. Далее, в соотвествии с ожидаемым оттоком, клиенты получают информацию об акции/скидке и других предложениях. Позже комадна DS проверяет результаты A/B теста.




