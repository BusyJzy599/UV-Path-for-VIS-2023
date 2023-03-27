/*
 * @Date: 2022-04-17 17:39:09
 * @LastEditors: JZY
 * @LastEditTime: 2023-01-02 18:57:53
 * @FilePath: /project/Visual/src/App.jsx
 */
import "./App.css";
import React, { Component } from "react";
import { Row, Layout, Space, Col, Spin, Alert } from "antd";
import axios from "axios";
// 导入组件
import CoreModule from "./components/CoreModule";
import MapVision from "./components/MapVision";
import LeftModule from "./components/LeftModule";
import TopModule from "./components/TopModule";

const { Content, Header } = Layout;

export default class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      choosePatches: [0],
      chooseMapImg: 0,
      mapValid: true,
      init_loading: 0,
      alert: {
        message: "Initial Progress",
        description:
          "Now, we have such 3 stage to do. First,we load the dataset.Then we pre-training model. Finally, we need generate AL result for 1th iteration...",
        type: "info",
      },
      epoch_Data: {},
      sample_Data: {},
      WSI_Data: {},
      confusion_Data: {},
      bk_data: {},
      current_iteration: 0,
    };
  }
  componentDidMount = () => {
    var T = this;
    axios
      .get("http://127.0.0.1:5000/init")
      .then(function (response) {
        if (response.data.load_status == 200) {
          T.setState({
            init_loading: 1,
            epoch_Data: response.data.epoch_Data,
            sample_Data: response.data.sample_data,
            WSI_Data: response.data.WSI_Data,
            confusion_Data: response.data.confusion_Data,
            bk_data: response.data.bk_data,
            current_iteration: response.data.iteration,
          });
        } else {
          T.setState({
            alert: {
              message: "Something error!",
              description:
                "We have a problem in initial stage, Please check your dataset...",
              type: "error",
            },
            init_loading: -1,
          });
        }
      })
      .catch((err) => {
        T.setState({
          alert: {
            message: "Connection error!",
            description:
              "We can't connect the data, Please check your network connection...",
            type: "error",
          },
          init_loading: -1,
        });
      });
  };
  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.coreModuleRef.changeDeletePatches(p);
  };
  changeChoosePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.mapChildRef.changeChoosePatches(p);
  };
  showMap = async (img_id) => {
    await this.setState({
      // mapValid: true,
      chooseMapImg: img_id,
    });
    this.mapChildRef.drawChart();
  };
  closeMap = () => {

    this.setState({
      mapValid: false,
    });
  };

  handleMapChildEvent = (ref) => {
    this.mapChildRef = ref;
  };
  handleCoreModuleEvent = (ref) => {
    this.coreModuleRef = ref;
  };

  render() {
    return (
      <>
        <Layout>
          {this.state.init_loading != 1 ? (
            <div id="example">
              <Spin spinning={this.state.init_loading == 0}>
                <Alert
                  message={this.state.alert.message}
                  description={this.state.alert.description}
                  type={this.state.alert.type}
                  showIcon
                />
              </Spin>
            </div>
          ) : (
            <>
              <Header className="headerModule" >
                <TopModule current_iteration={this.state.current_iteration} />
              </Header>
              <Content className="site-layout">
                <Space className="basic" direction="vertical" size="small">
                  <Row gutter={[5, 5]} justify="space-around" id="one">
                    <Col span={3}>
                      <LeftModule
                        epoch_Data={this.state.epoch_Data}
                        current_iteration={this.state.current_iteration}
                        confusion_Data={this.state.confusion_Data}
                      />
                    </Col>
                    <Col span={11} id="mainMap">
                      <CoreModule
                        onChildEvent={this.handleCoreModuleEvent}
                        ref={this.coreModuleRef}
                        changeChoosePatches={this.changeChoosePatches}
                        choosePatches={this.state.choosePatches}
                        mapValid={this.state.mapValid}
                        epoch_Data={this.state.epoch_Data}
                        sample_Data={this.state.sample_Data}
                        WSI_Data={this.state.WSI_Data}
                        bk_data={this.state.bk_data}
                        current_iteration={this.state.current_iteration}
                      />
                    </Col>
                    <Col span={10} id="mapVision">
                      <MapVision
                        onChildEvent={this.handleMapChildEvent}
                        ref={this.mapChildRef}
                        changeChoosePatches={this.changeChoosePatches}
                        showMap={this.showMap}
                        chooseMapImg={this.state.chooseMapImg}
                        changeDeletePatches={this.changeDeletePatches}
                        choosePatches={this.state.choosePatches}
                        epoch_Data={this.state.epoch_Data}
                        sample_Data={this.state.sample_Data}
                        WSI_Data={this.state.WSI_Data}
                        bk_data={this.state.bk_data}
                        current_iteration={this.state.current_iteration}
                      />
                    </Col>
                  </Row>
                </Space>
              </Content>
            </>
          )}
        </Layout>
      </>
    );
  }
}
