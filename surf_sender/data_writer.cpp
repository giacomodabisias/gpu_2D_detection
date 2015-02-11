#include "data_writer.h"

DataWriter::DataWriter(const std::string & uri): uri_(uri), m_open_(false), m_done_(false)
{
  m_client_.clear_access_channels(websocketpp::log::alevel::all);
  m_client_.set_access_channels(websocketpp::log::alevel::connect);
  m_client_.set_access_channels(websocketpp::log::alevel::disconnect);
  m_client_.set_access_channels(websocketpp::log::alevel::app);

  // Initialize the Asio transport policy
  m_client_.init_asio();
  std::cout << "connecting to " << uri_ <<std::endl;
  con_ = m_client_.get_connection(uri_, ec_);
  std::cout << "connection result " << ec_.message() << std::endl;
  m_hdl_ = con_->get_handle();
  m_client_.connect(con_);
  websocketpp::lib::thread asio_thread(&client::run, &m_client_);

}

void 
DataWriter::writeData(const std::string s)
{ 
  m_client_.send(m_hdl_, s , websocketpp::frame::opcode::text, ec_);
}


