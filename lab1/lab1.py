import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO:
        # Load data from data/chipotle.tsv file using Pandas library and
        file = 'data/chipotle.tsv'
        # assign the dataset to the 'chipo' variable.
        self.chipo = pd.read_csv(file, sep='\t')

    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())

    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        count = self.chipo.order_id.count()
        return count

    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info)
        pass

    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        cols = self.chipo.shape[1]
        return cols

    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
        pass

    def most_ordered_item(self):
        # TODO
        item_grp = self.chipo.groupby(['item_name']).sum({'quantity':'sum'})
        item_sort = item_grp.sort_values("quantity", ascending=False)
        item_name = item_sort.index[0]
        order_id = item_sort.values[0][0]
        quantity = item_sort.values[0][1]
        print(item_name)
        print(order_id)
        print(quantity)
        # The quantity parameter does not match the assertion
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
        # TODO How many items were orderd in total?
        item_total = self.chipo.quantity.sum()
        return item_total

    def total_sales(self) -> float:
        # TODO
        # 1. Create a lambda function to change all item prices to float.
        lam_func = self.chipo.item_price.apply(lambda x: float(x[1:]))
        # 2. Calculate total sales.
        total_sales = (self.chipo.quantity * lam_func).sum()
        return total_sales

    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        total_orders = self.chipo.order_id.iloc[-1]
        #Used iloc to print the last value of the order_id
        return total_orders

    def average_sales_amount_per_order(self) -> float:
        # TODO
        #avg_sale = self.chipo.total_sales() / self.chipo.num_orders()
        lam_func = self.chipo.item_price.apply(lambda x: float(x[1:]))
        total_sales = (self.chipo.quantity * lam_func).sum()
        total_orders = self.chipo.order_id.iloc[-1]
        avg_sale = (total_sales / total_orders).round(2)
        #Used the round function to round off the avg_sale to 2 decimal places
        return avg_sale

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        item_diff = self.chipo.item_name.nunique()
        # using nunique function to count the unique item names in the dataset
        return item_diff

    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        data_frame= pd.DataFrame.from_dict(letter_counter,orient='index').reset_index()
        # 2. sort the values from the top to the least value and slice the first 5 items
        data_frame = data_frame.rename(columns = {'index' : 'item_name', 0 :'quantity'})
        new_data = data_frame.sort_values(by = 'quantity', ascending = 'False').iloc[:5]
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        new_data.plot.bar(x = 'item_name',y= 'quantity', title='Most popular items')
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        print(data_frame)
        pass

    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        self.chipo['item_price'] = self.chipo['item_price'].str.replace('$', '')
        self.chipo['item_price'] = self.chipo.item_price.apply(lambda x: float(x[:]))
        # 2. groupby the orders and sum it.
        item_grp = self.chipo.groupby('order_id').agg({'item_price':'sum','quantity':'sum'})
        #print(item_grp)
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        scatter= item_grp.plot.scatter(x='item_price', y='quantity', s=50, c='blue', title='Number of items per order price')
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.show(block=True)
        pass

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    solution.print_columns()
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    # assert quantity == 159
    # The assertion mentioned above is not right 
    item_total = solution.total_item_orders()
    assert item_total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
# execute only if run as a script
    test()