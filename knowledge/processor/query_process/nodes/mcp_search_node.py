from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldEooror

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
import json
import asyncio
import os
import dotenv
dotenv.load_dotenv()


class McpSearchNode(BaseNode):
    """
    负责从网络上查询当前的问题（整个知识库都没有找到该问题，这个是兜底用的）
    mcp形式调用第三方的各种通用的搜索工具
    百度：【电商】商品比价工具、商品搜索的工具、商品全维度对比工具
    零积服务平台（百炼平台）
    mcp:本质就是各大平台吧通用功能封装为工具（函数） 然后通过mcp协议 客户端就可以直接俄调用它。
    """


    name = "mcp_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_entity_names = self._validate_query_inputs(state)

        # 2. 创建mcp客户端 并且执行工具调用
        mcp_result = asyncio.run(self._create_execute_web_search(validated_query))

        if not mcp_result:
            return state

        # 3. 更新state web_search_docs
        state['web_search_docs'] = mcp_result

        # 4. 返回更新后的state
        return  state
        
    

    def _validate_query_inputs(self, state: QueryGraphState) -> tuple[str, list[str]]:
        """校验查询输入参数。

        Args:
            state: 查询状态对象。

        Returns:
            校验后的查询字符串和商品名称列表。

        Raises:
            StateFieldEooror: 状态字段缺失或无效。
        """
        # 1. 获取state的rewritten_query字段
        rewritten_query = state.get("rewritten_query", "")

        # 2. entity_names
        entity_names = state.get("entity_names", "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldEooror(field_name="rewritten_query", node_name=self.name, expected_type = str)
        
        if not entity_names or not isinstance(entity_names, list):
            raise StateFieldEooror(field_name="entity_names", node_name=self.name, expected_type = list)
        
        # 4. 返回
        return rewritten_query, entity_names
    
    async def _create_execute_web_search(self, query: str) -> dict:
        """
        创建mcp客户端
        执行调用
        别人的接口: 1. 发送请求调用别人
        使用模型的客户端【1.原生发送请求2. mcp客户端发送请求3. Agent【也是mcp客户端发送请求】】
        Agent: 代理【代理对象就是程序员】
        Args:
            query: 要搜索的查询字符串。

        Returns:
            工具调用的结果字典。
        """
        try:
            # 1. 创建mcp客户端[1. host 2. http 2.1 streamable(主要建议使用)http 2.2sse http 3. stdio sse]
            mcp_client = MCPServerStreamableHttp(
                name = "通用搜索",
                cache_tools_list = True, # mcp服务端工具列表的工具做缓存
                params = MCPServerStreamableHttpParams({
                    "url": self.config.mcp_dashscope_base_url, # 服务端的端点
                    "headers":{"Authorization": self.config.bailian_api_key} # 认证权限 api_key
                },
                timeout= 10, # 超时时间 10秒
                sse_read_timeout= 60, # sse的超时时间 60秒
                )
            )
        except Exception as e:
            self.logger.error(f"创建MCP客户端失败: {e}")
            return []
        
        try:
            # 2. 建立mcp链接
            await mcp_client.connect()

            # 3. 执行工具
            tool_result = await mcp_client.call_tool(tool_name = "bailian_web_search", arguments = {"query": query, "count": 3})

            # 4. 解析工具执行完的结果
            # 4.1 获取最外层的对象
            if not tool_result:
                return []
            # 4.2 获取对象的content属性
            if not tool_result.content[0]:
                return []
            # 4.3  获取TextContent对象的text
            text_content_text: str = tool_result.content[0].text
            if not text_content_text:
                return []
            # 4.4 反序列化
            try:
                text_content_text: dict[str, any] = json.loads(text_content_text)

                # a) 获取pages
                pages = text_content_text.get('pages', "")
                if not pages:
                    return []
                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})
                # c) 最终返回
                return search_result
            except Exception as e:
                self.logger.error("反序列MCP结果失败")
                return []
        finally:
            await  mcp_client.cleanup()  # 关闭连接



if __name__ == '__main__':
    state = {
        'original_query': '18年的Ⅰ卷中“鲁芝字世英，扶风郿人也，世有名德，为西州豪族。”中的“也”有什么作用？',
        'rewritten_query': '请问“也”字在文言文句子“鲁芝字世英，扶风郿人也，世有名德，为西州豪族。”中有什么作用？',
        'entity_names': ['也']
    }

    mcp_search = McpSearchNode()

    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))

"""
{
  "snippet": "1. 用在句末,表示判断语气;2. 用在句末,表示陈述或解释语气;3. 用在句中,表示语气停顿;4. 用在句末,表示疑问语气;5. 用在句中或句末,表示肯定、感叹的语气;6. 用在句末,表示祈使语气;7. (也哉)语气助词连用,加强语气,多表感叹或反诘;8. (也者)语气助词连 用,起说明或解释作用。",
  "title": "“也”字在文言文中有什么意义?请说明其具体意思和用法。",
  "url": "https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1733383458356775512&fr=search"
}
{
  "snippet": "文言虚词“也”的用法如下: 1. 句末语气词: ①表示判断语气,如“张良曰:‘沛公之参乘樊哙者也。’(《鸿门宴》)”; ②表示陈述或解释语气,如“雷霆乍惊,宫车过也。(《阿房宫赋》)”; ③用在句中或句末,表示肯定、感叹的语气,如“灭六国者六国也,非秦也;族秦者秦也,非天下也。(《阿房宫赋》)”; ④用在句末,表示疑问或反诘语气,如“使秦复爱六国之人,则递三世可至万世而为君,谁得而族灭也?(《阿房宫赋》)”; ⑤用在句末,表示祈使语气,如“以乱易整,不武。吾其还也。(《烛之武退秦师》)”; 2. 句中语气词: 用在句中,表示语气停顿,如“其闻道也亦先 乎吾。(《师说》)”; 3. 固定词组: “也哉”语气助词连用,为加强语气,多有感叹或反诘之意,如“岂独伶人也哉?(《伶官传序》)”。",
  "title": "百度试题",
  "url": "https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1846656156957095356&fr=search"
}
{
  "snippet": "古文中“也”字主要作语气助词,用法包括:1. 表判断,用于句末,确认主语的性质或类别,如“陈胜者,阳城人也”;2. 表肯定或强调,加强语气,如“吾上恐负朝廷,下恐愧吾师也”;3. 表停顿,用于句中,舒缓语气,如“惩山北之塞,出入之迂也,聚室而谋曰”;4. 表感叹,表达强烈情感,如“悲哉,世也!”。注:“也”在古文中较少作副词,原提及的副词用法多为现代汉语用法,古文中“也”主要为语气助词。",
  "title": "请简述古文中“也”字的主要用法。",
  "url": "https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1733391330765146436&fr=search"
}
"""