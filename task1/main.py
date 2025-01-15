from collections import deque

def is_new_island(map_details, row_index, column_index):
    map_state = map_details[row_index][column_index]
    return map_state == 1

def add_neighbor_nodes(nodes, map_details, row_index, column_index):
    rows_amount, columns_amount = len(map_details), len(map_details[0])

    if row_index > 0 and map_details[row_index - 1][column_index] == 1:
        nodes.append((row_index - 1, column_index))
    if column_index + 1 < columns_amount and map_details[row_index][column_index + 1] == 1:
        nodes.append((row_index, column_index + 1))
    if row_index + 1 < rows_amount and map_details[row_index + 1][column_index] == 1:
        nodes.append((row_index + 1, column_index))
    if column_index > 0 and map_details[row_index][column_index - 1] == 1:
        nodes.append((row_index, column_index - 1))

    return nodes

def remove_island_from_map(map_details, row_index, column_index):
    nodes = deque([(row_index, column_index)])
    
    while nodes:
        row_index, column_index = nodes.popleft()
        map_details[row_index][column_index] = 0
        nodes = add_neighbor_nodes(nodes, map_details, row_index, column_index)
    
    return map_details

def main():
    rows, columns = map(int, input().split())
    map_details = [list(map(int, input().split())) for _ in range(rows)]

    islands_amount = 0
    for row_index in range(rows):
        for column_index in range(columns):
            if not is_new_island(map_details, row_index, column_index):
                continue
            islands_amount += 1
            map_details = remove_island_from_map(map_details, row_index, column_index)

    print(islands_amount)

if __name__ == "__main__":
    main()
