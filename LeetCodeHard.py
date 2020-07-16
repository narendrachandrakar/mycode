class Solution:
    def findWords(self, board, words):
        ## RC ##
        ## APPROACH : TRIE + DFS ##
        ## LOGIC ##
        #	1. build trie with all words list
        #	2. start scanning matrix, move in all four directions and check if such combination exists in the trie
        #	3. make sure you donot return when result is found ( case: words = [cat, cattle ] )

        ## TIME COMPLEXICITY : O(M(4x3^(L-1))) ## (M is the number of cells in the board and L is the maximum length of words.)
        ## SPACE COMPLEXICITY : O(N) ##

        def dfs(i, j, curr, currNode):
            ch = board[i][j]
            if (ch not in currNode.children or (i, j) in visited):
                return
            if currNode.children[ch].endOfWord:
                res.add(curr)
                # return                            # edge case
            visited.add((i, j))
            for x, y in directions:
                if 0 <= i + x < m and 0 <= j + y < n:
                    dfs(i + x, j + y, curr + board[i + x][j + y], currNode.children[ch])
            visited.remove((i, j))  # edge case

        # buid trie data structure
        my_trie = Trie()
        [my_trie.insert(word) for word in words]
        rootNode = my_trie.get_rootNode()

        m, n = len(board), len(board[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        res = set()
        for i in range(len(board)):
            for j in range(len(board[i])):
                visited = set()
                dfs(i, j, board[i][j], rootNode)
        return res


class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False


class Trie:
    def __init__(self):
        self.rootNode = TrieNode()

    def get_rootNode(self):
        return self.rootNode

    # Inserts a word into the trie.
    def insert(self, word: str) -> None:
        currNode = self.rootNode
        for idx, ch in enumerate(word):
            if (ch not in currNode.children):
                currNode.children[ch] = TrieNode()
            currNode = currNode.children[ch]
        currNode.endOfWord = True


input = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
words = ["oath","pea","eat","rain"]

obj = Solution()
#print(obj.findWords(input, words))

# ++++++++++++++++++++++++++++++++++++++++++++++++++







