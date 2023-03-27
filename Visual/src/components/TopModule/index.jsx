/*
 * @Date: 2022-05-14 20:45:48
 * @LastEditors: JZY
 * @LastEditTime: 2023-02-05 16:26:33
 * @FilePath: /project/Visual/src/components/TopModule/index.jsx
 */
import React, { Component } from "react";
import { Typography, Row, Col, Select, Button, Tooltip, Tag } from "antd";
import {
  GithubOutlined,
  QuestionCircleOutlined,
  ShareAltOutlined,
  SyncOutlined
} from "@ant-design/icons";

import "./index.css";

const { Option } = Select;
const { Text, Title } = Typography;

const info = [
  ["Dataset:", ["peso",'hubmap'], 0],
  ["WSI Count:", ["30"], 0],
  ["Backbone:", ["ResNet50"], 0],
  ["Scatter:", ["UMAP", "TSNE"], 0],

];

export default class TopModule extends Component {
  constructor(props) {
    super(props);
  }
  toGithub = () => {
    window.open("https://github.com/BusyJzy599")
  }

  render() {
    return (
      <>
        <Row gutter={8} align="top" >
          <Col span={2}>
            <Title>UV-Path</Title>
          </Col>

          {info.map((item, index) => {
            return (
              <>
                <Col>
                  <Text className="siderFont">{item[0]}</Text>
                </Col>
                <Col>
                  <Select defaultValue={item[2]} size="small">
                    {
                      item[1].map((e, idx) => {
                        return (
                          <Option value={idx}>{e}</Option>
                        )

                      })
                    }
                  </Select>
                </Col>
              </>
            );
          })}
          <Col>
            <Tag icon={<SyncOutlined spin />} color="processing">
              Training  Epoch  {this.props.current_iteration+1}
            </Tag>
          </Col>
          <Col offset={11}>
            <Tooltip placement="bottomRight" title="Click the icon to jump to Github page.">
              <Button type="text" className="siderFont" shape="circle" icon={<GithubOutlined />} onClick={this.toGithub} />
            </Tooltip>
            <Tooltip placement="bottomRight" title="The doi of our paper is ...">
              <Button type="text" className="siderFont" shape="circle" icon={<ShareAltOutlined />} />
            </Tooltip>
            <Tooltip placement="bottomRight" title="If you have some problems,please contact us via 1411020952@qq.com.">
              <Button type="text" className="siderFont" shape="circle" icon={<QuestionCircleOutlined />} />
            </Tooltip>

          </Col>
        </Row>

      </>
    );
  }
}
